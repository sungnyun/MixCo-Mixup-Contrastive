import os, torch
from models import *
from data_utils import *
from train_tools import *
from eval_tool import *


MODEL = {'resnet10': resnet10, 'resnet18': resnet18, 'resnet50': resnet50,
         'ResSimclr': ResNetSimCLR}

CRITERION = {'CE': nn.CrossEntropyLoss,
             'NTXent': NTXentLoss}

OPTIMIZER = {'sgd': optim.SGD}

SCHEDULER = {'multistep': lr_scheduler.MultiStepLR,
            'cosine': lr_scheduler.CosineAnnealingLR}


def _get_dataset(opt):
    args = opt.dataset
    dataloaders, dataset_sizes = data_loader(args.dataset.params.__dict__)
    
    return dataloaders, dataset_sizes


def _get_model(opt):
    args = opt.model
    device = opt.train.device
    
    model = Model[args.type]
    model = model(**args.param.__dict__)
    model.to(device)
    
    #if args.pretrained.enabled:
    #    model.load_state_dict(torch.load(args.pretrained.path), map_location=device)
        
    return model
        
    

def _get_train(opt):    
    args = opt.train
    
    model = _get_model(opt)

    criterion = CRITERION[args.criterion.algo]
    criterion = criterion(**args.criterion.param.__dict__)
    
    optimizer = OPTIMIZER[args.optimizer.algo]
    optimizer = optimizer(model.parameters(), **args.optimizer.param.__)
    
    return model, criterion, optimizer, scheduler


def main():
    # fix random seeds
    torch.manual_seed(opt.train.seed)
    np.random.seed(opt.train.seed)
    torch.backends.cudnn.deterministic = True
    
    # dataloaders
    dataloaders, dataset_sizes = _get_dataset(opt)
    
    # model
    model, criterion, optimizer, scheduler, = _get_train(opt)
    
    if opt.trainer.use_wandb:
        wandb.init(project=opt.experiment_info.project_name, 
                   name=opt.experiment_info.name, 
                   tags=opt.experiment_info.tags,
                   group=opt.experiment_info.group,
                   notes=opt.experiment_info.notes, 
                   config=configdict)
        wandb.watch(model, log="all")
        wandb.config.log_interval = 10     # how many batches to wait before logging training status
    
    save_path = path_setter('./results', opt.experiment_info.name, '')
    directory_setter(save_path, make_dir=True)

    # Save test result
    torch.save(model.state_dict(), os.path.join(save_path, 'model.h5'))
    
    wandb.save(os.path.join(save_path, 'model.h5'))

if __name__ == '__main__':
    configdict['train']['seed'] = args.seed
    configdict['train']['device'] = args.device
    opt = objectview(configdict)


    
    ######## WandB – Config is a variable that holds and saves hyperparameters and inputs ###########
    
    # training settings
    
    main()
