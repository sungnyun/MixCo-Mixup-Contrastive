#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="/home/osilab/dataset/cifar10"
exp_name="mixco_resnet18_cifar10to10"

# pretrain encoder
python main.py -a resnet18 --builder mixco --batch-size 128 --lr 0.015 --epochs 100 --schedule 60 80 --dataset cifar10 --data-path $data_path --mlp --moco-t 0.2 --aug-plus --cos --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10003

# linear classification protocol
python lincls.py -a resnet18 --lr 3.0 --epochs 100 --dataset cifar10 --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10003
