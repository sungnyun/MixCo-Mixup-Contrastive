#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="/home/osilab/dataset/tiny-imagenet-200"
exp_name="mixco_ver4_resnet50"

# pretrain encoder
python main.py -a resnet50 --builder mixco --batch-size 128 --lr 0.015 --epochs 200 --schedule 120 160 --dataset tiny-imagenet --data-path $data_path --mlp --moco-t 0.2 --aug-plus --cos --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001

# linear classification protocol
python lincls.py -a resnet50 --lr 3.0 --epochs 100 --schedule 60 80 --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001
