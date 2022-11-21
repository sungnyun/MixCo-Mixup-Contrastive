#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="../data/tiny-imagenet-200"
exp_name="exp_simclr_res18_gpu2_tinyimg"

# pretrain encoder
# python ./pretrain.py -a resnet18 --batch-size 256 --epochs 1 --dataset tiny-imagenet --data-path $data_path --temperature 0.5 --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1

# linear classification protocol
python ./lincls.py -a resnet18 --lr 3.0 --epochs 100 --schedule 60 80  --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}/checkpoint.pth --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 
