# treina modelo unico
from sklearn.model_selection import KFold
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib import pyplot as plt
from datetime import datetime
from adasampling import AdaSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from classifiers import ClassificationModel
import numpy as np

def train_single_model(dm, model_name='resnet18'):
    moment = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    
    hp = [0.0001, 0.9, 0.0001, 10]
    
            
    count_classes = [len(dm.busbra_train[dm.busbra_train['Pathology'] == 'benign']), len(dm.busbra_train[dm.busbra_train['Pathology'] == 'malignant'])]
    class_weights = 1/torch.tensor(count_classes)
    class_weights = class_weights/torch.mean(class_weights)
        
    model_hparams={"in_channels": 3, "num_classes": 2, "act_fn_name": "relu"}
    optimizer_name="SGD"
    optimizer_hparams={"lr": hp[0], "momentum": hp[1], "weight_decay": hp[2]}
    max_epochs = 150
    patience = 20
    
    early_stop_callback = EarlyStopping(
                            monitor="val_loss", 
                            patience=patience, 
                            verbose=False, 
                            mode="min"
                            )

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        precision='16-mixed',
        max_epochs=max_epochs,
        callbacks = [early_stop_callback, 
                     # checkpoint_callback
                    ],
        accumulate_grad_batches=2,
        num_sanity_val_steps=0
    )

    model = ClassificationModel(model_name, model_hparams, optimizer_name, 
                                optimizer_hparams, 
                                loss_weight=class_weights
                               ) 
    
    trainer.fit(
        model=model, 
        datamodule=dm
        )
    
    model.sweights, model.alpha = update_adaboost(dm, model)
    
    val_metrics = trainer.validate(model=model, datamodule=dm,
                                  verbose=False)
    del trainer
    return model

def set_trainer():      
    max_epochs = 150
    patience = 20
    
    early_stop_callback = EarlyStopping(
                            monitor="val_loss", 
                            patience=patience, 
                            verbose=False, 
                            mode="min"
                            )

    trainer = pl.Trainer(
                accelerator="gpu", 
                devices=1, 
                precision='16-mixed',
                max_epochs=max_epochs,
                callbacks = [early_stop_callback, 
                            ],
                accumulate_grad_batches=2,
                num_sanity_val_steps=0
            )
    return trainer

from  torch.cuda.amp import autocast
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error
import torch.nn as nn


class MyAdaBoostEnsemble(pl.LightningModule):
    def __init__(self,
                models, loss_weight=None,
                with_doctor=False
                ):
        super(MyAdaBoostEnsemble, self).__init__()
        self.models = models
        self.cuda0 = torch.device('cuda:0')
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.weights = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        self.with_doctor = with_doctor

        for i in range(len(self.models)):
            self.models[i].freeze()
            self.models[i].to(self.cuda0)
                        
        self.loss_module = nn.CrossEntropyLoss(weight=loss_weight)
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        if self.with_doctor:
            birads = x[1]
            x = x[0]
        x0 = torch.nn.functional.softmax(self.models[0](x) * self.models[0].alpha)  # logits * alpha

        for i in range(1,len(self.models)):
            xi = torch.nn.functional.softmax(self.models[i](x) * self.models[i].alpha)  # logits * alpha
            x0 = x0 + xi  # soma dos logits ponderados

        # com medico 
        if self.with_doctor:
            x0 = x0 + birads
            x0 = x0/(len(self.models)+1)
        else:
            x0 = x0/len(self.models)

        return x0
    
    def shared_step(self, batch, stage):
        imgs, labels = batch
        
        with autocast():
            preds = self.forward(imgs)
            loss = self.loss_module(preds, labels)
                
        outputs = {
            "loss": loss,
            "labels": labels,
            "preds": preds
        }
        
        if stage == 'train':
            self.training_step_outputs.append(outputs) #detach? 

        return outputs
    
    def shared_epoch_end(self, outputs, stage):
        labels = torch.cat([x["labels"] for x in outputs]).cpu()
        probs = torch.cat([x["preds"] for x in outputs]).cpu()  # logits softmax já aplicados no forward
        preds = (probs[:, 1] > 0.457).long()
        loss = torch.cat([x["loss"].reshape(1) for x in outputs]).cpu()
        
        acc = (preds == labels).float().mean()
        loss = loss.mean()
        
        if stage == 'test':
        
            cm = confusion_matrix(labels, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['benign', 'malignant'])
            disp.plot(cmap='Blues')

            plt.show()
        
        metrics = {
             f"{stage}_acc": acc,
             f"{stage}_f1_score": f1_score(labels, preds),
             f"{stage}_sens": recall_score(labels, preds),
             f"{stage}_spec": recall_score(labels, preds, pos_label=0),
             f"{stage}_loss": loss            
         }
        
        self.log_dict(metrics, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs.copy()
        self.training_step_outputs.clear()
        return self.shared_epoch_end(outputs,"train")

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self.shared_step(batch, "val"))
        return self.shared_step(batch, "val")
    
    def on_validation_epoch_end(self):
         return self.shared_epoch_end(self.validation_step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, 'test')

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)

def update_adaboost(dm, new_model):
    err = np.sum(new_model.is_wrong.numpy()*dm.sweights)/dm.sweights.sum()
    err = np.clip(err, 1e-10, 1 - 1e-10)
    # adicionando divisao por 2 para diminuir o peso
    new_alpha = 1/2*np.log((1 - err)/err) + np.log(new_model.hparams['model_hparams']['num_classes']-1)
    new_sweights = dm.sweights * np.exp(new_alpha*new_model.is_wrong.numpy())
    new_sweights = new_sweights/new_sweights.sum()

    return new_sweights, new_alpha

def evaluate_ensemble_step(dm, ensemble_list, new_models):
    print('chamou evaluate ensemble step')
    ensemble_f1 = 0
    for model in new_models:
        trainer = set_trainer()
        temp_ensemble_list = ensemble_list.copy()
        
        temp_ensemble_list.append(model)
        # criar temp_ensemble
        temp_ensemble = MyAdaBoostEnsemble(temp_ensemble_list)
        val_metrics = trainer.validate(model=temp_ensemble, datamodule=dm,
                                      verbose=False)
        if val_metrics[0]['val_f1_score'] > ensemble_f1:
            ensemble_f1 = val_metrics[0]['val_f1_score']
            new_model = model
            best_val_metrics = val_metrics
        del model
        del trainer
    print(f'val do ensemble eh {val_metrics}')

    print(f'sweights antigo eh: {dm.sweights}')
    print(f'obs que o modelo errou: {new_model.is_wrong}')
    dm.sweights = new_model.sweights
    print(f'sweights novo eh: {dm.sweights}')
    ensemble_list.append(new_model)
    
    return ensemble_list, best_val_metrics