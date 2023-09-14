"""
Train a diffusion model on images.
"""
import sys
import os
sys.path.append(os.path.realpath('./'))

import argparse
import pathlib
from guided_diffusion import dist_util, logger
from data import get_data_iter
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from sample import sample


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log(f"args: {args}")
    
    args.num_classes = int(args.num_classes) if args.num_classes is not None else None
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # get model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    logger.log('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    pathlib.Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    if args.name.lower() == 'brats':
        kwargs = dict(
            n_healthy_patients=int(args.n_healthy_patients) if args.n_healthy_patients is not None else None, 
            n_tumour_patients=int(args.n_tumour_patients) if args.n_tumour_patients is not None else None,
            mixed=args.mixed)
    else:
        kwargs = dict()
   
    
    data = get_data_iter(args.name, 
                         args.data_dir, 
                         args.batch_size,
                         split=args.split,
                         ret_lab=args.ret_lab,
                         logger=logger,
                         training=args.training,
                         kwargs=kwargs) 
    


    logger.log("training...")
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_shape=tuple(args.sample_shape),
        img_dir=args.image_dir,
        threshold=args.threshold,
        w=args.w,
        num_classes=args.num_classes,
        name = args.name,
        sample_fn=sample
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        name="",
        split="train",
        training=True,
        mixed=False,
        ret_lab=False,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        n_tumour_patients=None,
        n_healthy_patients=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_shape",
        type=int,
        nargs="+",
        help="sample shape"
    )
    
    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1. # disabled in default
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold for clf-free training",
        default=-1. # disabled in default
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
