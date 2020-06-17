import argparse
import yaml

parser = argparse.ArgumentParser()

# Configurations
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=None, nargs='+')
parser.add_argument('--num-workers', type=int, default=8)

# Training Options
parser.add_argument('-d', '--dataset', type=str, default='imagenet', help='imagenet|cifar10|cifar100')
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('--probe-freq', type=int, default=5)


args = parser.parse_args()



if args.dataset == 'imagenet':
    with open('path_data.yaml') as f:
        path_data = yaml.safe_load(f)
    args.dir_data = path_configs['dir_imagenet']


