from models.SimCLR import SimCLR
from trainers import *
from eval_tools.lossfuncs import NTXentLoss
from data_utils import data_loader
from utils.utils import set_device
from config import args
from torch.nn.parallel import DistributedDataParallel

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist


model = SimCLR(base_model='resnet50', out_dim=args.proj_dim, from_small=True)

if args.distributed:
    ngpus_per_node = torch.cuda.device_count()
    nodes = 1  # default, 1 machine
    args.world_size = ngpus_per_node * nodes

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(train, nprocs=ngpus_per_node, args=(args,))
else:
    train(None, args)

def train(gpu, args):
    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.distributed:
        dist_url="env://"
        rank = gpu
        dist.init_process_group(backend='nccl',
                                init_method=dist_url,
                                world_size=args.world_size,
                                rank=rank)
        device = set_device(gpu)
        model = DistributedDataParallel(model.to(device))
    else:
        device = set_device(args)
        model = nn.DataParallel(model.to(device))

    dataloaders, dataset_sizes = data_loader(args.dataset, 
                                             args.dir_data, 
                                             rep_augment='simclr', 
                                             batch_size=args.batch_size,
                                             valid_size=args.valid_size,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             drop_last=True,
                                             distributed=args.distributed,
                                             num_replicas=args.world_size,
                                             rank=rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    # TODO: check batch size -> in distributed setting, may be divided
    criterion = NTXentLoss(device, args.batch_size, args.temperature, use_cosine_similarity=True)

    trainer = SimCLRTrainer(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device)

    probe_loaders, probe_sizes = data_loader(args.dataset,
                                             args.dir_data,
                                             rep_augment=None,
                                             batch_size=128)
    # TODO: change probe_setup
    probe_setup = {'dataloaders': probe_loaders,
                   'dataset_sizes': probe_sizes,
                   'num_classes': 1000}

    trainer.train(num_epochs=args.epochs, save_path=args.save_dir, phase='pretrain')
    #trainer.test(save_path, do_probe=True, probe_setup=probe_setup)
