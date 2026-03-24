import torch
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
from math import log

class LoadDataset(Dataset):
    def __init__(self, df, transform=None, pred=False, dataset_name=None, images_names=False):
        self.df = df
        self.transform = transform
        self.pred = pred
        self.dataset_name = dataset_name
        self.images_names = images_names
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_name = self.df['ID'][idx]
        label = self.df['Pathology'][idx]
        
        if self.dataset_name == 'busi':
            mask_name = image_name.replace('.png', '_mask.png')
            image_path = '../Dataset_BUSI_with_GT/' + label + '/' + image_name
            mask_path = '../Dataset_BUSI_with_GT/' + label + '/' + mask_name

        if self.dataset_name == 'breast':
            path = '../BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/'
            mask_name = image_name.replace('.png', '_tumor.png')
            image_path = path + image_name
            mask_path = path + mask_name
            
        img = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        bbox = self.df['BBOX'][idx].replace('[', ' ').replace(']', ' ').replace(',', ' ')
        bbox = [int(s) for s in bbox.split() if s.isdigit()]
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        
        img = img.crop([x_min-5, y_min-5, x_min+w+5, y_min+h+5])
        mask = mask.crop([x_min-5, y_min-5, x_min+w+5, y_min+h+5])
        
        from skimage.exposure import rescale_intensity
        img = rescale_intensity(np.array(img),
            in_range=tuple(np.percentile(np.array(img), (5, 95))))
 
        mask = np.array(mask)
        pred_seg = img*(mask/255)
        fusion_image = np.array([np.asarray(img).astype(np.uint8), pred_seg.astype(np.uint8), mask.astype(np.uint8)]).transpose(1,2,0)

        fusion_image = Image.fromarray(np.uint8(fusion_image))
        fusion_image = fusion_image.convert('RGB')
        
        label = 1 if label=='malignant' else 0
        if self.transform:
            fusion_image = self.transform(fusion_image)
        if self.images_names:
            return fusion_image, label, image_name
        if not self.pred:
            return fusion_image, label
        else:
            return fusion_image

class BUSIDataModule(pl.LightningDataModule):
    def __init__(self, df, dataset_name=None, shuffle=False, images_names=False):
        super().__init__()
        self.df = df
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.images_names = images_names

        self.transform = {
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        }

    def test_dataloader(self):
        test_dataset = LoadDataset(self.df, self.transform['val'], dataset_name=self.dataset_name, 
                                   images_names=self.images_names)
        return DataLoader(test_dataset, batch_size=32, shuffle=self.shuffle,
                         )
        
    def predict_dataloader(self):
        pred_dataset = LoadDataset(self.df, self.transform['val'], 
                                   dataset_name=self.dataset_name,pred=True, images_names=self.images_names)
        return DataLoader(pred_dataset, batch_size=32, shuffle=False, 
                          num_workers=14, persistent_workers=True)

    def imshow_img(self, inp, title=None):
        plt.rcParams['figure.figsize'] = [30, 20]

        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) 

    def imshow_test(self):
        if self.images_names:
            images, labels, images_names = next(iter(self.test_dataloader()))
        else:
            images, labels = next(iter(self.test_dataloader()))

        out = torchvision.utils.make_grid(images)
        print(labels)

        self.imshow_img(out)