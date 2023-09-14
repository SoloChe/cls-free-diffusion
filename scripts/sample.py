# adapted from https://github.com/SoloChe/ddib/blob/main/scripts/synthetic_sample.py

import argparse
import os
import sys
sys.path.append(os.path.realpath('./'))

import pathlib

import numpy as np
import torch as th
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import  model_and_diffusion_defaults, add_dict_to_argparser

from torchvision.utils import make_grid, save_image


def sample(model, diffusion, num_classes=None, 
           w=None, sample_shape=None, name=None):
    samples_for_each_cls = 5 # default
    if num_classes is not None: # for clf-free
        samples_for_each_cls = sample_shape[0] // num_classes
        y = th.ones(samples_for_each_cls, dtype=th.long) *\
            th.arange(start = 0, end = num_classes).reshape(-1, 1)
        y = y.reshape(-1,1).squeeze().to(dist_util.dev())
        model_kwargs = {'y': y}
    else:
        model_kwargs = {}
        
    samples = diffusion.ddim_sample_loop(model, 
                                         sample_shape,
                                         clip_denoised=False, 
                                         w=w,
                                         model_kwargs=model_kwargs,
                                         device=dist_util.dev())
        
    if name == 'cifar10': 
        samples = (samples + 1) / 2 # normalize it back
    return samples, samples_for_each_cls

def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure()
    logger.log(f"args: {args}")
    logger.log("starting to sample.")
    
    args.num_classes = int(args.num_classes) if args.num_classes is not None else None
    
    image_folder = args.image_dir
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    model, diffusion = read_model_and_diffusion(args, args.model_path)

    all_samples = []

    
    for n in range(args.n_batches):
        logger.log("sampling in progress.")
        logger.log(f"on batch {n}, device: {dist_util.dev()}")
        
        # save samples
        
        
        samples, samples_for_each_cls = sample(model, diffusion, num_classes=args.num_classes, 
                         w=args.w, sample_shape=args.sample_shape, name=args.name)
        
        if args.save_image:
            grid = make_grid(samples, nrow=samples_for_each_cls)
            path = os.path.join(image_folder, f'batch_{n}_rank_{dist.get_rank()}.png')
            save_image(grid, path)
        
        if args.save_data_numpy:
            gathered_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples)
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

    if args.save_data_numpy:
        samples = np.concatenate(all_samples, axis=0)
        points_path = os.path.join(image_folder, f"all_samples.npy")
        np.save(points_path, samples)
    

    dist.barrier()
    logger.log(f"sampling synthetic data complete\n\n")


def create_argparser():
    defaults = dict(
        n_batches=4,
        model_path="",
        image_dir="",
        name="",
        save_data_numpy=True, # save data in numpy format
        save_image=True # save image as in png format
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--sample_shape",
        type=int,
        nargs="+",
        help="sample shape for a batch"
    )
    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1. # disabled in default
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
