# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np


class MixCo(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.07, alpha=1.0, eps=0.0, mlp=False):
        super(MixCo, self).__init__()
        self.T = T
        self.alpha = alpha
        self.eps = eps
        self.encoder = base_encoder(num_classes=dim)
        
        if mlp:
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)

    def forward(self, im_a, im_b):
        """
        Input:
            im_a: a batch of aug1
            im_b: a batch of aug2
        Output:
            logits, targets
        """
        
        im_mix, lam, labels = self.data_mixer(im_a, im_b, alpha=self.alpha, eps=self.eps)
        
        # compute features
        z1 = self.encoder(im_a)
        z1 = nn.functional.normalize(z1, dim=1) # 256 * 128
        
        z_mix = self.encoder(im_mix)
        z_mix = nn.functional.normalize(z_mix, dim=1) # 256 * 128
        
        # compute logits
        logits = torch.mm(z1, z_mix.T) # 256 * 256
        logits /= self.T 

        return logits, labels

    @torch.no_grad()
    def data_mixer(self, x_aug1, x_aug2, alpha=1.0, eps=0.0):
        # batch size
        b = x_aug1.shape[0]

        idx1 = torch.Tensor(range(b)).cuda()
        idx2 = torch.randperm(b).cuda()

        # mixup process
        lam = torch.Tensor(np.random.beta(alpha, alpha, size=b)).cuda()
        lam = lam.reshape(b,1,1,1)
        x_aug2 = x_aug2[idx2]    # shuffle samples
        x_mix = lam*x_aug1 + (1-lam)*x_aug2
        
        x_aug2.cpu()

        lam = lam.reshape(b,1) # dim b
        target1 = onehot(idx1.long(), N=b)
        target1 = eps*(torch.ones(b).cuda()/b) + (1-eps) * target1
        target2 = onehot(idx2.long(), N=b)
        target2 = eps*(torch.ones(b).cuda()/b) + (1-eps) * target2
        
        ins_label = lam*target1 + (1-lam)*target2

        return x_mix, lam, ins_label

    
def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output