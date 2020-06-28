from models.SimCLR import SimCLR
from trainers import *
from eval_tools.lossfuncs import NTXentLoss
from data_utils import data_loader
from utils.utils import set_device
from config import args

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


device = set_device(args)
model = SimCLR(base_model='resnet50', out_dim=args.proj_dim, from_small=True)
if torch.cuda.device_count() >= 2:
    model = nn.DataParallel(model)

dataloaders, dataset_sizes = data_loader(args.dataset, 
                                         args.dir_data, 
                                         rep_augment='simclr', 
                                         batch_size=args.batch_size,
                                         valid_size=args.valid_size,
                                         pin_memory=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
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

trainer.train(num_epochs=args.epochs, phase='pretrain')
#trainer.test(save_path, do_probe=True, probe_setup=probe_setup)
