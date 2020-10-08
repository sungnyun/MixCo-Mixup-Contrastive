#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="/home/osilab/dataset/tiny-imagenet-200"
exp_name="exp_simclr_res34_gpu4_batch128_tinyimg"

# pretrain encoder
python ./pretrain.py -a resnet34 --batch-size 128 --lr 0.15 --epochs 200 --dataset tiny-imagenet --data-path $data_path --temperature 0.5 --exp-name $exp_name --cos --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 2 3 --out-dim 128 --save-freq 10

# linear classification protocol
python ./lincls.py -a resnet34 --lr 3.0 --batch-size 256 --epochs 100 --schedule 60 80  --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 2 3 --wd 0.0 
