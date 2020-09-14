import torch, os
from umap import UMAP
import matplotlib.pyplot as plt


__all__ = ['top5_correct', 'feature_concater', 'visualizer', 'directory_setter', 'path_setter']


def top5_correct(output, target):
    """
    Computes the precision@k for the specified values of k
    """
    batch_size = target.size(0)

    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
    correct_5 = correct[:5].view(-1).float().sum(0, keepdim=True)

    return correct_1, correct_5


def feature_concater(model, dataloaders, phases=['test'], device='cuda:0', from_encoder=True):
    if type(phases) == str:
        phases = [phases]
    
    model.eval()
    model.to(device)
    stacked_feature = torch.Tensor([]).to('cpu')
    stacked_labels = torch.Tensor([]).long().to('cpu')
                    
    with torch.no_grad():
        for phase in phases:
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                if from_encoder:
                    features, _ = model(inputs) # from h(representation), x(output after linear)
                else:
                    _, features = model(inputs) # when inferenced from a normal network (e.x. ResNet)
                    
                stacked_feature = torch.cat((stacked_feature, features.to('cpu')))
                stacked_labels = torch.cat((stacked_labels, labels.to('cpu')))

        return stacked_feature.cpu(), stacked_labels.cpu()

    
def visualizer(features, labels, save_path='./results', name='test'):
    save_path = path_setter(save_path, name, 'umap')
    directory_setter(save_path, make_dir=True)

    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    reducer = UMAP()
    embedding = reducer.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    x, y = embedding[:,0], embedding[:,1]

    for i in range(len(self.classes)):
        y_i = self.labels == i
        plt.scatter(x[y_i], y[y_i], label=self.classes[i], alpha=0.7)

    plt.legend(fontsize=15)
    fname = os.path.join(save_path, 'umap_vis.png')
    plt.savefig(fname)
    plt.close()

    return save_path, fig
            
    
def directory_setter(path='./results', make_dir=False):
    if not os.path.exists(path) and make_dir:
        os.makedirs(path) # make dir if not exist
        print('directory %s is created' % path)
        
    if not os.path.isdir(path):
        raise NotADirectoryError('%s is not valid. set make_dir=True to make dir.' % path)

        
def path_setter(result_path, sub_loc, model_name):
    save_path = '/'.join((result_path, sub_loc, model_name))
    return save_path
