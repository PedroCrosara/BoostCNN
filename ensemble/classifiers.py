from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import classification_report
import torchmetrics
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, f1_score, fbeta_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

import timm

def cm_analysis(cm_init, names, ymap=None, figsize=(17,17)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2.8)

    cm = cm_init
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = cm_init
    cm = pd.DataFrame(cm, index=names, columns=names)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=names, cbar=False, cbar_kws={'format':PercentFormatter()}, yticklabels=names, cmap="Blues")
    plt.show()
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# função chamada dentro do ClassificationModel para criar o modelo a partir de model_name
# exemplo na resnet18
def create_model(model_name, in_channels, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        # cria o modelo resnet18 com os pesos pré-treinados do imagenet
        model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # para transfer learning, feature_extract=False
        # para extração de características, feature_extract=True
        # no segundo caso, as camadas convolucionais são congeladas e 
        # apenas as camadas completamente conectadas (linear) são treinadas 
        set_parameter_requires_grad(model_ft, feature_extract)
        # como os pesos carregados são referentes à imagenet, que possui 10 classes 
        # de saída, precisamos trocar a última camada linear para uma com a quantidade
        # n de classes do nosso problema. Normalmente 2 classes
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wide-resnet":
        """ Wide Resnet
        """
        model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vit":
        """ Visual Transformers
        """
        model_ft = models.vit_b_16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == "swint":
        """ Swin Transformer
        """
        model_ft = models.swin_v2_s(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
        
    elif model_name == "mobilenet_small":
        """ MobileNet
        """
        model_ft = models.mobilenet_v3_small(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
        model_ft.num_classes = num_classes
        input_size = 224
        
    elif model_name == "mobilenet_large":
        """ MobileNet
        """
        model_ft = models.mobilenet_v3_large(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet201(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "convnext":
        model_ft = timm.create_model('convnext_tiny.fb_in1k',
                                     pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.fc.in_features
        model_ft.head.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.num_classes = num_classes
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, loss_weight=None):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters()
        # listas que armazenam as saídas do modelo em cada etapa
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # 1 se errou e 0 se acertou
        self.is_wrong = None
        self.class_probs = None
        self.train_labels = None

        self.alpha = None
        self.sweights = None

        # dicionário para plotar gráfico de f1xloss do modelo
        self.history = {'train_f1_score': [],
                        'train_loss':[],
                        'val_f1_score': [],
                        'val_loss': []
                        }
        
        # Create model
        self.model = create_model(model_name, model_hparams["in_channels"], 
                                  model_hparams["num_classes"], False, use_pretrained=True)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss(weight=loss_weight)
        
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    # step do modelo que ocorre em cada batch 
    # tanto na etapa de treino, quanto na de validação e de teste
    def shared_step(self, batch, stage):
        # recebe as imagens do datamodule
        imgs, labels = batch
        # passa as imagens pelo modelo
        preds = self.model(imgs)
        # calcula a loss do modelo
        loss = self.loss_module(preds, labels)
                
        outputs = {
            "loss": loss,
            "labels": labels,
            "preds": preds,
        }

        # salva os outputs de treino de cada batch, para que possam 
        # ser utilizados no fim do treinamento 
        if stage == 'train':
            self.training_step_outputs.append(outputs)
        return outputs

    # função que é chamada no fim de cada época (note que não no fim de cada batch)
    def shared_epoch_end(self, outputs, stage):
        # descompacta as labels, os preds e a loss de todos os batches
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach()
        preds = torch.cat([x["preds"] for x in outputs]).cpu().detach().argmax(dim=-1)
        model_output = torch.cat([x["preds"] for x in outputs]).cpu().detach()
        loss = torch.cat([x["loss"].reshape(1) for x in outputs]).cpu()

        # calcula métricas
        acc = (preds == labels).float().mean()
        loss = loss.mean()
        sens = recall_score(labels, preds)
        spec = recall_score(labels, preds, pos_label=0)
        f1 = f1_score(labels, preds)
        loss_hist = loss.detach()

        # para treino e validação limpa os outputs para a próxima época e
        # dá o append nas métricas para o plot
        if stage == 'train':
            self.is_wrong = (preds!=labels).float()
            self.class_probs = torch.nn.functional.softmax(model_output, dim=1)
            self.train_labels = labels
            self.training_step_outputs.clear()
            self.history['train_f1_score'].append(f1)
            self.history['train_loss'].append(loss_hist)

        if stage == 'val':
            self.validation_step_outputs.clear()
            self.history['val_f1_score'].append(f1)
            self.history['val_loss'].append(loss_hist)

        # para o teste fazemos o classification_report, plotamos
        # uma matriz de confusão e limpamos os outputs
        if stage == 'test':
            cm = confusion_matrix(labels, preds)

            cm_analysis(cm, ['benign', 'malignant'], ymap=None, figsize=(13,13))
            
            self.test_step_outputs.clear()

        metrics = {
             f"{stage}_acc": acc,
             f"{stage}_sens": sens,
             f"{stage}_spec": spec,
             f"{stage}_loss": loss,
             f"{stage}_f1_score": f1,   
         }
        
        self.log_dict(metrics, prog_bar=True)

    # função que chama a shered_step para cada batch de treinamento
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # função que chama o shared_epoch_end no fim de uma época de treinamento
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

    # otimizadores que são utilizados para atualizar os pesos do modelo
    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            return optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        
        if self.hparams.optimizer_name == "SGD":
            return optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)