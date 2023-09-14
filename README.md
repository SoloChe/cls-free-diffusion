# Implementation of Classifier-free Diffusion Model on cifar10
This implementation is based on / inspired by:
OpenAI: [openai/guided-diffusion](https://github.com/openai/guided-diffusion), [openai/improved-diffusion](https://github.com/openai/improved-diffusion) and [ddib](https://github.com/suxuann/ddib).

<!-- <img src="assets/15000.png" height="240" /> -->

## The main modifications
the function `condition_clf_free` is added in the script `./guidied_diffusion/gaussian_diffusion.py`. The `ddim` sampling loop can call `condition_clf_free` for classifier-free guidance. Note that `ddpm` sampling is not modified yet. The original implementation of classifier-guided sampling is still working.

`torchrun` is used to replace the `mpirun` for convenience. Some modifications about `torchrun` are added in `./guidied_diffusion/dist_utill.py`

## Usage
The scripts for running unguided training `run_train_cifar10.sh` and guided training `run_train_cifar10_guided.sh` are provided. It will save model and sample during training every `--save_interval`. You also need to specify `--data_dir`.

The defauslt parameter `--num_class` is set to `None` and `--class_cond` to `False` for unguided training. For classifier-free guided training, please specify `--num_class`, `--w` and`--threshold` with `--class_cond True` for `train.py`. 
- `--num_class`: the number of classes, same as the original implementation 
- `--threshold`: the ratio of data that their label are embedded as null embeddings
- `-w`: the guidance power during sampling (It will sample during training)

For classifier-guided sampling, it is the same as the original implementation, except that set `--w -1`, `--threshold -1` and `class_cond True`. 

The model and sampled images will be in `./logs_cifar10_guided_$threshold`.

For other details, please refer to the aforementioned repositories.

The scripts `./scripts/sample.py` and `run_sample.sh` for sample from pre-trained model are also provided. You need to specify `--model_path`, e.g., `--model_path ./logs_cifar10_guided_0.1`. It will read from the newest model in the path.