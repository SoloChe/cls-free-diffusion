#!/bin/bash

export OPENAI_LOGDIR="./logs_sample_guided"
echo $OPENAI_LOGDIR


model_path="./logs_cifar10_guided_0.1"

image_dir="$OPENAI_LOGDIR/images"

# These are the flags for guided sampling
# for guided sampling, specify --num_classes and --w and set --class_cond True
# for unguided sampling, set --class_cond False

SAMPLE_FLAGS="--num_classes 10 --w 1.8 --class_cond True --timestep_respacing ddim100 \
                    --n_batches 4 --sample_shape 100 3 32 32"

diffusion_steps=1000
DIFFUSION_FLAGS="--diffusion_steps $diffusion_steps --noise_schedule linear \
                    --rescale_learned_sigmas False --rescale_timesteps False"

DIR_FLAGS="--image_dir $image_dir --model_path $model_path"

MODEL_FLAGS="--image_size 32 --in_channels 3 --num_channels 64 --attention_resolutions 1 --dropout 0.1"

NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS ./scripts/sample.py --name cifar10 $MODEL_FLAGS $DIFFUSION_FLAGS $DIR_FLAGS $SAMPLE_FLAGS $MODEL_FLAGS