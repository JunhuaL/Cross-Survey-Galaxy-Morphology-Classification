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
from dataset import Galaxy10_Dataset

if __name__ == '__main__':
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transforms = [sys.argv[1],sys.argv[2]]
    print("applying these data augmentations: ",transforms)

    root_folder = 'outputs/'
    save_model_folder = 'outputs/models/' + '+'.join(transforms) + '/'

    datamodule = Galaxy10_Dataset('Galaxy10.h5')

    # model = SimCLR_container(3,1024,transforms_list=transforms)

    # earlystopping_tracking = 'trn_ntxent_loss'
    # earlystopping_mode = 'min'
    # earlystopping_min_delta = 0.0001

    # checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=save_model_folder,
    #                                                    mode = earlystopping_mode,
    #                                                    monitor=earlystopping_tracking,
    #                                                    save_top_k=1,save_last=True,)
    
    # earlystop_callback = pl_callbacks.EarlyStopping(earlystopping_tracking,verbose=True,
    #                                     mode = earlystopping_mode,
    #                                     min_delta=earlystopping_min_delta,
    #                                     patience=10,)

    # trainer = Trainer(
    #                 gpus=[0,],
    #                 accelerator=None,
    #                 max_epochs=200, min_epochs=5,
    #                 default_root_dir= root_folder,
    #                 fast_dev_run=False,
    #                 check_val_every_n_epoch=1,
    #                 callbacks=  [checkpoint_callback,
    #                             earlystop_callback,],
    #                 )
    # trainer.fit(model, datamodule=datamodule,)

    model_file = save_model_folder+f'simCLR.{transforms[0]}.{transforms[1]}.pt'
    #t.save(model.model.state_dict(),model_file)


    lin_Eval = LightningDSModel(3,1024,model_file,10,False,0.001)

    earlystopping_tracking = 'train_loss'
    earlystopping_mode = 'min'
    earlystopping_min_delta = 0.0001

    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=save_model_folder,
                                                       mode = earlystopping_mode,
                                                       monitor=earlystopping_tracking,
                                                       save_top_k=1,save_last=True,)
    
    earlystop_callback = pl_callbacks.EarlyStopping(earlystopping_tracking,verbose=True,
                                        mode = earlystopping_mode,
                                        min_delta=earlystopping_min_delta,
                                        patience=10,)

    trainer = Trainer(
                    gpus=[0,],
                    accelerator=None,
                    max_epochs=200, min_epochs=5,
                    default_root_dir= root_folder,
                    fast_dev_run=False,
                    check_val_every_n_epoch=1,
                    callbacks=  [checkpoint_callback,
                                earlystop_callback,],
                    )

    trainer.fit(lin_Eval,datamodule)

    model_file = save_model_folder+f'DSModel.{transforms[0]}.{transforms[1]}.pt'
    t.save(lin_Eval.model.state_dict(),model_file)