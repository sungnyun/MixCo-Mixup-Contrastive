import os

import torch


def set_device(args):
    if args.device is not None:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        device = str(args.device[0])
        for i in range(len(args.device) - 1):
            device += (',' + str(args.device[i+1]))
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    return torch.device('cuda')

