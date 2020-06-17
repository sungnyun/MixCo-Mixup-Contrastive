import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=None, nargs='+')
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--dataset', type=str, default='imagenet', help='imagenet|cifar10|cifar100')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()
