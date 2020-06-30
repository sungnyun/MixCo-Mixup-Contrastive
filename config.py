import argparse
import yaml

parser = argparse.ArgumentParser()

# Configurations
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=None, nargs='+')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--dir-data', type=str, default='./data') 
parser.add_argument('--save-dir', type=str, default='./save/test')
parser.add_argument('--distributed', action='store_false', default=True)

# Training Options
parser.add_argument('-d', '--dataset', type=str, default='imagenet', help='imagenet|cifar10|cifar100')
parser.add_argument('-m', '--model', type=str, default='resnet50', help='resnet10|18|50')
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--proj-dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--valid-size', type=int, default=5000)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('--probe-freq', type=int, default=None)


args = parser.parse_args()



if args.dataset == 'imagenet':
    with open('path_data.yaml') as f:
        path_config = yaml.safe_load(f)
    args.dir_data = path_config['dir_imagenet']

# LARS
if args.lr == -1:
    args.lr = 0.3 * args.batch_size / 256

print(args)
