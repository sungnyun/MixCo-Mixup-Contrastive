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
dist_url+=:10003

DATASET_PATH="../data/tiny-imagenet-200"
EXPERIMENT_PATH="./experiments/swav_200ep_bs128_gpu2_lincls"
mkdir -p $EXPERIMENT_PATH

python -u eval_linear.py \
--data_path $DATASET_PATH \
--pretrained './experiments/swav_200ep_bs128_gpu2_pretrain/checkpoint.pth.tar' \
--dist_url $dist_url \
--arch resnet18 \
--dump_path $EXPERIMENT_PATH \
--multiprocessing-distributed \
--gpu 0 1 
