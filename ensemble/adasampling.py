import pytorch_lightning as pl
import numpy as np
import torch

class AdaSampler(pl.Callback):
    def __init__(self, patience=1):
        self.patience = patience
        self.num_classes = None
        self.is_wrong = None
        self.class_probs = None
        self.train_labels = None
        self.sweights = None
        
    def calculate_sweights(self, trainer, model):
        if (trainer.current_epoch+1) % self.patience == 0:
            self.sweights = trainer.datamodule.sweights
            self.is_wrong = model.is_wrong
            self.class_probs = model.class_probs
            self.train_labels = model.train_labels
            self.num_classes = model.hparams['model_hparams']['num_classes']

            self.update_sweights()
            if max(self.sweights) == 0:
                trainer.should_stop = True
            else:
                trainer.datamodule.update_sweights(self.sweights)
                
    def update_sweights(self, method='real'):
        if self.is_wrong is not None:
            if method == 'discrete':
                self.discrete_adaboost()
            elif method == 'real':
                self.real_adaboost()
                
    def discrete_adaboost(self):
        err = np.sum(self.is_wrong.numpy()*self.sweights)/self.sweights.sum()
        err = np.clip(err, 1e-10, 1 - 1e-10)
        alpha = np.log((1 - err)/err) + np.log(self.num_classes-1)
        self.sweights = self.sweights * np.exp(alpha*self.is_wrong.numpy())
        self.sweights = self.sweights/self.sweights.sum()

    def real_adaboost(self):
        pmx = self.class_probs[:,1]
        fmx = 1/2 * torch.log(pmx/(1 - pmx + 1e-8))
        fmx = fmx.detach().numpy()
        yi = (2*self.train_labels - 1).numpy()

        # atualiza wi
        self.sweights = self.sweights * np.exp(-yi*fmx) 
    
        # re-normaliza
        self.sweights = self.sweights/self.sweights.sum()
            
    def on_train_epoch_start(self, trainer, model):
        return self.calculate_sweights(trainer, model)