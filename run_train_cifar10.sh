#!/bin/bash

export OPENAI_LOGDIR="./logs_cifar10"
echo $OPENAI_LOGDIR


data_dir="/home/local/ASUAD/yche14/project/DATA/cifar10"
image_dir="$OPENAI_LOGDIR/images"

DATA_FLAGS="--image_size 32 --class_cond False --ret_lab False"

MODEL_FLAGS="--in_channels 3 --num_channels 64 --attention_resolutions 1"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

TRAIN_FLAGS="--data_dir $data_dir --image_dir $image_dir --batch_size 128 --dropout 0.1"

EVA_FLAGS="--save_interval 5000 --sample_shape 50 3 32 32"




NUM_GPUS=1
torchrun --nproc-per-node $NUM_GPUS ./scripts/train.py --name cifar10 $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $EVA_FLAGS