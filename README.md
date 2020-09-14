## How to Reproduce our Results

This is an instruction to reproduce our results, based on the source code we have provided.

### Prerequisites

1. You should download the Tiny-ImageNet dataset. To download the images, go to [https://tiny-imagenet.herokuapp.com/](https://tiny-imagenet.herokuapp.com/) and click 'Download Tiny ImageNet' button. Equivalently, try
```sh
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip 
```
2. Unzip the file and place the folder into `[your_data_path]`.
3. For linear classification, you may also need CIFAR10 and CIFAR100 dataset. If you do not have them, find the line below and change to `download=True`. Then, it will download the dataset before training.
```sh
train_dataset = DATASETS[args.dataset](args.data_path, train=True, download=False, transform=train_transform)
``` 

### Experiments

1. In `./experiments/` directory, there are `.sh` files which include the commands that can reproduce our experimental results. Open and set the configs.
```sh
data_path="[your_data_path]"
exp_name="[experiment_name]"
```
2. Run the file. For example, if you want to pretrain the ResNet18 model with Tiny-ImageNet, and then see the linear classification results, run `exp_mix_res18.sh`.
```sh
CUDA_VISIBLE_DEVICES=[GPUs] bash experiments/exp_mix_res18.sh
```

