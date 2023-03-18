import sys
import os
import warnings
import numpy as np
import pandas as pd
import yaml
warnings.filterwarnings('ignore')
import argparse
import torch as t 
from torch import Tensor
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from model import SimCLR_container,LightningDSModel
from dataset import Galaxy10_Dataset, GalaxyZooUnlabelled_dataset

if __name__ == '__main__':
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transforms = sys.argv[1:]
    print("applying these data augmentations: ",transforms)

    root_folder = 'outputs/final/' + '+'.join(transforms) + '/'
    save_model_folder = root_folder + 'models/'

    datamodule = Galaxy10_Dataset('Galaxy10.h5')
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        X,y = batch
        for img in X:
            print(img.dtype)