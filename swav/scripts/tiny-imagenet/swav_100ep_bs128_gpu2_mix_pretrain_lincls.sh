# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=swav_200ep_bs256_pretrain
#SBATCH --time=72:00:00
#SBATCH --mem=150G

#master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+="localhost"
dist_url+=:10001

DATASET_PATH="../data/tiny-imagenet-200"
EXPERIMENT_PATH="./experiments/swav_resnet18_100ep_bs128_gpu2_mix_pretrain"
mkdir -p $EXPERIMENT_PATH

python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 64 \
--nmb_prototypes 400 \
--queue_length 1200 \
--epoch_queue_starts 10 \
--epochs 100 \
--batch_size 128 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 300 \
--wd 0.000001 \
--warmup_epochs 0 \
--dist_url $dist_url \
--arch resnet18 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH \
--multiprocessing-distributed \
--gpu 0 1 \
--mix_temperature 0.1 \
--mix 


EXPERIMENT_PATH="./experiments/swav_resnet18_100ep_bs128_gpu2_mix_lincls"
mkdir -p $EXPERIMENT_PATH

python -u eval_linear.py \
--data_path $DATASET_PATH \
--pretrained './experiments/swav_resnet18_100ep_bs128_gpu2_mix_pretrain/checkpoint.pth.tar' \
--dist_url $dist_url \
--arch resnet18 \
--dump_path $EXPERIMENT_PATH \
--multiprocessing-distributed \
--gpu 0 1
