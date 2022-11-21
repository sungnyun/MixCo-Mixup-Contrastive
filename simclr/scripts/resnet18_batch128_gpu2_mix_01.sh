#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="/data/home/bsmn0223/data/tiny-imagenet-200"
exp_name="exp_simclr_res18_gpu2_batch128_mix_01_tinyimg"

# pretrain encoder
python ./pretrain.py -a resnet18 --batch-size 128 --lr 0.15 --epochs 100 --dataset tiny-imagenet --data-path $data_path --temperature 0.5 --exp-name $exp_name --cos --multiprocessing-distributed --dist-url tcp://localhost:10002 --gpu 2 3 --out-dim 128 --save-freq 10 --mix --mix-temperature 0.1

# linear classification protocol
python ./lincls.py -a resnet18 --lr 3.0 --batch-size 256 --epochs 100 --schedule 60 80  --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10002 --gpu 2 3 --wd 0.0 
