import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim 
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning.losses import NTXentLoss
import torchvision.transforms as transforms
from typing import Optional
from pytorch_lightning import LightningDataModule
import h5py

class GaussianNoise:
    def __init__(self,mean=0.,std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self,tensor):
        return tensor.cuda() + torch.cuda.FloatTensor(tensor.size()).normal_() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean},std={self.std})'
    

transforms_dict = {'crop+resize': transforms.RandomApply([transforms.RandomResizedCrop(size=96)],p=0.5),
                   'colorjitter': transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.1),
                   'gray': transforms.RandomApply([transforms.RandomGrayscale(p=0.2)],p=0.5),
                   'blur': transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)],p=0.5),
                   'rotation': transforms.RandomRotation(degrees=(0,360)),
                   'gauss_noise': transforms.RandomApply([GaussianNoise(mean=0, std=0.05)],p=0.5)}

class Galaxy10_Dataset(LightningDataModule):
    def __init__(self,datadir,batch_size = 32,dataNumPerClass = None):
        super(Galaxy10_Dataset).__init__()
        self.file_path = datadir
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        self._log_hyperparams = True
        self.dataCount = dataNumPerClass

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        with h5py.File(self.file_path, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])
        
        labels = np.eye(10)[labels]
        labels = labels.astype(np.float16, copy=False)
        images = images.astype(np.float16, copy=False)
        images = images/255
        images = images.transpose((0,3,1,2))

        images = torch.from_numpy(images).share_memory_()
        labels = torch.from_numpy(labels).share_memory_()
        X_train, X_test, y_train, y_test = train_test_split(images,labels, test_size = 0.2)
        # X_train, y_train = getBalanceDataset(X_train,y_train,self.dataCount,['colorjitter','rotation','gauss_noise'])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.1)

        self.train = TensorDataset(X_train,y_train)
        self.valid = TensorDataset(X_valid,y_valid)
        self.test = TensorDataset(X_test,y_test)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size = 32, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = 32, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size = 32, shuffle=True)

class GalaxyZooUnlabelled_dataset(LightningDataModule):
    def __init__(self, datadir, batch_size = 32):
        super(GalaxyZooUnlabelled_dataset).__init__()
        self.file_path = datadir
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        self._log_hyperparams = True

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        dataset = torch.load('dataset_final.pt').share_memory_()
        dataset = torch.permute(dataset, (0,3,1,2))

        labels = torch.zeros(dataset.size(0)).share_memory_()

        X_train, X_test, y_train, y_test = train_test_split(dataset,labels, test_size = 0.001)
        X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, test_size = 0.5)

        self.train = TensorDataset(X_train,y_train)
        self.valid = TensorDataset(X_valid,y_valid)
        self.test = TensorDataset(X_test,y_test)

        print(X_train.shape)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size = 8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = 8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = 8, shuffle=True)



t2np = lambda t: t.detach().cpu().numpy()
def getBalanceDataset(X,y,dataCount,transforms_list):
    X = t2np(X)
    y = t2np(y)
    data_aug = transforms.Compose([transforms_dict[transform] for transform in transforms_list])
    
    if(len(y.shape) == 2):
        labels = y.argmax(axis=1)
    if(dataCount == None):
        # Get the second small class count
        labelCounts = np.unique(labels, return_counts=True)[1]
        labelCounts.sort()
        dataCount = labelCounts[1]
    
    dataCountDic = [dataCount] * y.shape[1]
    
    X_balanced = []
    y_balanced = []
    
    for i in range(len(y)):
        image = X[i]
        label = int(labels[i])
        if(dataCountDic[label] > 0):
            X_balanced.append(image)
            y_balanced.append(label)
            
            dataCountDic[label] = dataCountDic[label] - 1
    
    up_sampled_imgs = []
    up_sampled_labels = []
    for i,quota in enumerate(dataCountDic):
        sample_idxs = labels==i
        sample_images = X[sample_idxs]
        sample_labels = labels[sample_idxs]
        up_sample_idxs = np.random.randint(0,len(sample_images)-1,quota)
        up_sample = sample_images[up_sample_idxs]
        up_sample_labels = np.eye(10)[sample_labels[up_sample_idxs]]
        up_sampled_imgs.append(up_sample)
        up_sampled_labels.append(up_sample_labels)
        
    up_sampled_imgs = torch.from_numpy(np.concatenate(up_sampled_imgs)).cuda()
    up_sampled_labels = torch.from_numpy(np.concatenate(up_sampled_labels)).cuda()
    up_sampled_imgs = torch.from_numpy(np.array([t2np(data_aug(img)) for img in up_sampled_imgs])).cuda()
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    y_balanced = np.eye(10)[y_balanced]
    X_balanced = torch.from_numpy(X_balanced).cuda()
    y_balanced = torch.from_numpy(y_balanced).cuda()
    
    return torch.cat((X_balanced,up_sampled_imgs),0), torch.cat((y_balanced,up_sampled_labels),0)