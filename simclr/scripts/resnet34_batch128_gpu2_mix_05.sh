#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="/home/osilab/dataset/tiny-imagenet-200"
exp_name="exp_simclr_res34_gpu2_batch128_mix_05_tinyimg"

# pretrain encoder
python ./pretrain.py -a resnet34 --batch-size 128 --lr 0.15 --epochs 200 --dataset tiny-imagenet --data-path $data_path --temperature 0.5 --exp-name $exp_name --cos --multiprocessing-distributed --dist-url tcp://localhost:10003 --gpu 0 1 --out-dim 128 --save-freq 50 --mix --mix-temperature 0.5

# linear classification protocol
python ./lincls.py -a resnet34 --lr 3.0 --batch-size 256 --epochs 100 --schedule 60 80  --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10003 --gpu 0 1 --wd 0.0 

python ./lincls.py -a resnet34 --lr 3.0 --batch-size 256 --epochs 100 --schedule 60 80 --dataset cifar10 --data-path /home/osilab/dataset/cifar10 --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name simclr_res34_mix_05_tinyimgtocifar10 --multiprocessing-distributed --dist-url tcp://localhost:10003 --gpu 0 1 --wd 0.0

python ./lincls.py -a resnet34 --lr 3.0 --batch-size 256 --epochs 100 --schedule 60 80 --dataset cifar100 --data-path /home/osilab/dataset/cifar100 --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name simclr_res34_mix_05_tinyimgtocifar100 --multiprocessing-distributed --dist-url tcp://localhost:10003 --gpu 0 1 --wd 0.0
