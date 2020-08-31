#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="./data/tiny-imagenet"
gpu=1
exp_name="mixco_ver2"

# pretrain encoder
python ./main.py -a resnet18 --builder mixco --batch-size 128 --lr 0.015 --epochs 100 --schedule 60 80 --dataset tiny-imagenet --data-path $data_path --mlp --moco-t 0.2 --aug-plus --cos --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001

# linear classification protocol
python ./lincls.py -a resnet18 --lr 3.0 --epochs 100 --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --gpu $gpu --exp-name $exp_name
