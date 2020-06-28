# Refactored from https://github.com/sthalles/SimCLR

import torch
import numpy as np
import torch.nn.functional as F
from .NTXentLoss import NTXentLoss


__all__ = ['SoftNTXentLoss']

class SoftNTXentLoss(NTXentLoss):

    def __init__(self, device, batch_size=128, temperature=1, use_cosine_similarity=True, decay_type='base'):
        super(SoftNTXentLoss, self).__init__(device, batch_size, temperature, use_cosine_similarity)

    def forward(self, zis, zjs, epoch=None):
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        
        loss = self.criterion(logits, labels)
        pred = self.prediction(logits)
        measure = torch.sum(pred == labels.data) / 2.
        
        #return loss / (2 * self.batch_size), measure
        return loss, measure
    
    def _decay_planer(self):
        