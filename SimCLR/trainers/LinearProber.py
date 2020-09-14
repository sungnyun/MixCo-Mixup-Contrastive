import torch, copy
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .utils import *

__all__ = ['LinearProber']


class LinearProber():
    """Test representation encoder performance via linear classification"""
    def __init__(self, encoder, dataloaders, dataset_sizes, device='cuda:0', \
                 num_classes=100, rep_channel_dim=512, pool_size=1, pool_type='avg'):
        """
        rep_channel_dim : dim of representation dimension
        pool_size : if the rep dim is [N * rep_channel_dim * M * M], apply pool_size = M
        pool_type : either 'avg' or 'max'
        """
        
        self.prober = LinearClassifier(num_classes, rep_channel_dim, pool_size, pool_type)
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()

        self.dataloaders, self.dataset_sizes = dataloaders, dataset_sizes
        self.optimizer = optim.SGD(self.prober.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = None
        self.device = device
        
        self.prob_loader = dict()
        self._data_setter(device)
    
    
    def train(self, num_epochs=20, decay_milestone=[12, 16]):
        print('========= Linear Probe Training =========')
        
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=decay_milestone, gamma=0.2)
        best_model_wts = copy.deepcopy(self.prober.state_dict())
        best_acc = 0.0
        
        for epoch in range(1, num_epochs+1):
            train_loss, train_acc, _ = self._prober_epoch_phase(phase='train')
            valid_loss, valid_acc_1, valid_acc_5 = self._prober_epoch_phase(phase='valid')
            
            self._print_stat(epoch, num_epochs, train_loss, train_acc, valid_loss, valid_acc_1, valid_acc_5)
            
            if valid_acc_1 > best_acc:
                best_acc = valid_acc_1
                best_model_wts = copy.deepcopy(self.prober.state_dict())
        
        self.prober.load_state_dict(best_model_wts)
            
        return train_loss, train_acc, valid_loss, valid_acc_1, valid_acc_5
    
    
    def test(self, get_vis=False):
        test_loss, test_acc_1, test_acc_5 = self._prober_epoch_phase(phase='test')
        print('========= Linear Probe Test Results =========')
        print(('[Test] Loss - {:.4f}, Top1 Acc - '.format(test_loss)) + ('{:2.2f}%'.format(test_acc_1*100)))
        print('[Test] Top5 Acc - ' + ('{:2.2f}%'.format(test_acc_5*100)))
        print('=============================================')
        
        if get_vis:
            ### umap visualization. To be implemented.
            pass
        
        return test_loss, test_acc_1, test_acc_5
    

    def _prober_epoch_phase(self, phase):
        if phase == 'train':
            self.prober.train()
        else:
            self.prober.eval()
            
        running_loss, running_correct_1, running_correct_5  = 0.0, 0.0, 0.0
        
        for i, (inputs, labels) in enumerate(self.prob_loader[phase]):
            self.optimizer.zero_grad()
        
            with torch.set_grad_enabled(phase == 'train'):
                # forward pass
                logits = self.prober(inputs)
                loss = self.criterion(logits, labels)

                # backward pass
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

                # prediction
                correct_1, correct_5 = top5_correct(logits, labels)

                # update statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_1 += torch.sum(correct_1).item()
                running_correct_5 += torch.sum(correct_5).item()
                
        if phase == 'train':
            self.scheduler.step()

        epoch_loss = round(running_loss / self.dataset_sizes[phase], 4)
        epoch_acc_1 = round(running_correct_1 / self.dataset_sizes[phase], 4)
        epoch_acc_5 = round(running_correct_5 / self.dataset_sizes[phase], 4)
        
        return epoch_loss, epoch_acc_1, epoch_acc_5

    
    def _print_stat(self, epoch, num_epochs, train_loss, train_acc, valid_loss, valid_acc_1, valid_acc_5):
        print('[Linear Probe Epoch {}/{}]'.format(epoch, num_epochs))
        print(('[{}] Loss - {:.4f}, Top1 Acc - '.format('Train', train_loss)) + '{:2.2f}% '.format(train_acc*100))
        print(('[{}] Loss - {:.4f}, Top1 Acc - '.format('Valid', valid_loss)) + ('{:2.2f}% '.format(valid_acc_1*100)))
        print(('[{}] Top5 Acc - '.format('Valid')) + ('{:2.2f}% '.format(valid_acc_5*100)))
        print('-' * 45)
        
    
    def _data_setter(self, device):
        """load all the representations for datasets. will be used as dataloaders for training classifier."""
        for phase in ['train', 'valid', 'test']:
            features, labels = feature_concater(self.encoder, self.dataloaders, phase, device, from_encoder=True)
            prob_set = ProbDataset(features, labels)
            self.prob_loader[phase] = torch.utils.data.DataLoader(prob_set, batch_size=256, num_workers=5)
        
    
class LinearClassifier(nn.Module):
    def __init__(self, num_classes=100, channel_dim=512, pool_size=1, pool_type='avg'):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Sequential()
        
        if pool_type == 'avg':
            self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
        else:
            self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        
        self.classifier.add_module('Flatten', nn.Flatten())
        self.classifier.add_module('LiniearClassifier', nn.Linear(channel_dim * pool_size * pool_size, num_classes))
        
        self._initilize()


    def forward(self, x):
        if len(x.shape) != 4:
            x = x.reshape(x.size(0), x.size(1), 1, 1)
        
        output = self.classifier(x)
        return output

    
    def _initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

                
class ProbDataset(Dataset):
    """
    Prob Dataset for loading from feature & label tensors.
    """

    def __init__(self, features, labels):
        super(ProbDataset, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
