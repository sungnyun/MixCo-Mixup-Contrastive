from .BaseTrainer import BaseTrainer

class SupTrainer(BaseTrainer):
    def __init__(self, model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, use_wandb=False):
        super(SupTrainer, self).__init__(model, dataloaders, dataset_sizes, criterion, 
                                            optimizer, scheduler, device, use_wandb)
        self.measure_name = 'Acc'
        self.valid_type='max_measure'
        
        
    def _step(self, inputs, labels):
        # gpu loading
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # inference
        outputs = self._inference(inputs)
        logits, features = outputs
        
        # *batch_size since assuming mean loss
        loss = self.criterion(logits, labels) * inputs.size(0)
            
        # prediction & count correct
        pred = logits.max(dim=1)[1]
        measure = (pred == labels).sum() * 100 # *100 to make as percentile
        
        return loss, measure
        
    def _inference(self, inputs):
        outputs = self.model(inputs)
        return outputs
