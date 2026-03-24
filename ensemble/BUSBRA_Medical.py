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
    def __init__(self, dataset_df, transform=None, return_birads=False, pred=False):
        self.dataset_df = dataset_df
        self.transform = transform
        self.pred = pred
        self.return_birads = return_birads
        
    def __len__(self):
        return self.dataset_df.shape[0]
    
    def __getitem__(self, idx):
        image_name = self.dataset_df['ID'][idx]
        mask_name = image_name.replace('bus', 'mask')
        image_path = '../BUSBRA/Images/' + image_name + '.png'
        mask_path = '../BUSBRA/Masks/' + mask_name + '.png'
        img = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        bbox = self.dataset_df['BBOX'][idx].replace('[', ' ').replace(']', ' ').replace(',', ' ')
        bbox = [int(s) for s in bbox.split() if s.isdigit()]
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]

        birads = self.dataset_df['BIRADS'][idx]
        
        img = img.crop([x_min, y_min, x_min+w, y_min+h])
        mask = mask.crop([x_min, y_min, x_min+w, y_min+h])
        
        #contrast stretching
        from skimage.exposure import rescale_intensity
        img = rescale_intensity(np.array(img),
            in_range=tuple(np.percentile(np.array(img), (5, 95))))

        mask = np.array(mask)
        pred_seg = img*(mask/255)
        fusion_image = np.array([np.asarray(img).astype(np.uint8), pred_seg.astype(np.uint8), mask.astype(np.uint8)]).transpose(1,2,0)

        fusion_image = Image.fromarray(np.uint8(fusion_image))
        fusion_image = fusion_image.convert('RGB')
        
        label = self.dataset_df['Pathology'][idx]
        label = 1 if label=='malignant' else 0
        
        birads = self.dataset_df['BIRADS'][idx]
        if birads == 2:
            birads = torch.tensor([0.9, 0.0])
        elif birads == 3:
            birads = torch.tensor([0.6, 0.4])
        elif birads == 4:
            birads = torch.tensor([0.4, 2*0.6])
        elif birads == 5:
            birads = torch.tensor([0.0, 6*0.9])

        if self.transform:
            fusion_image = self.transform(fusion_image)
        if self.return_birads:
            if not self.pred:
                return [fusion_image, birads], label
            else:
                return [fusion_image, birads]
        else:
            if not self.pred:
                return fusion_image, label
            else:
                return fusion_image


class AdaDataModule(pl.LightningDataModule):
    def __init__(self, train=None, val=None, test=None, birads=False):
        super().__init__()

        self.busbra_train = train
        self.busbra_val = val
        self.busbra_test = test
        self.sweights = None
        self.birads= birads

        self.transform = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation([-22.5,22.5]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        }

    def setup(self, stage):        

        if self.sweights is None:
            self.sweights = np.ones(len(self.busbra_train))/len(self.busbra_train)

    def train_dataloader(self):
        train_dataset = LoadDataset(dataset_df=self.busbra_train, 
                                    transform=self.transform['train'], 
                                    return_birads=self.birads
                                   )
        sampler = WeightedRandomSampler(weights=self.sweights, 
                                        replacement=True, 
                                        num_samples=len(train_dataset))
        return DataLoader(train_dataset, 
                      sampler=sampler, 
                      batch_size=32, 
                      shuffle=False, 
                      num_workers=14, 
                      persistent_workers=True) 
    
    def val_dataloader(self):
        val_dataset = LoadDataset(dataset_df=self.busbra_val, 
                                  transform=self.transform['val'], 
                                  return_birads=self.birads
                                 )
        return DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=14, persistent_workers=True)

    def test_dataloader(self):
        test_dataset = LoadDataset(dataset_df=self.busbra_test, 
                                   transform=self.transform['val'], 
                                   return_birads=self.birads
                                  )
        return DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=14)

    def update_sweights(self, new_sweights):
        self.sweights = new_sweights
            
    def predict_dataloader(self):
        pred_dataset = LoadDataset(dataset_df=self.busbra_test, 
                                   transform=self.transform['val'], 
                                   return_birads=self.birads, 
                                   pred=True
                                  )
        return DataLoader(pred_dataset, batch_size=32, shuffle=False, num_workers=14)

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

    def imshow_train(self):
        images, labels = next(iter(self.train_dataloader()))

        out = torchvision.utils.make_grid(images)
        print(labels)

        self.imshow_img(out)

    def imshow_val(self):
        images, labels = next(iter(self.val_dataloader()))

        out = torchvision.utils.make_grid(images)
        print(labels)

        self.imshow_img(out)

    def imshow_test(self):
        images, labels = next(iter(self.test_dataloader()))

        out = torchvision.utils.make_grid(images)
        print(labels)

        self.imshow_img(out)