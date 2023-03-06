import torch
import os
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from autoEncoder import DSModelLightning,AutoencoderLightning
from dataset import Galaxy10_Dataset


latent_size = 128
learning_rate = 0.01
root_folder = 'outputs/'
save_model_folder = root_folder + 'models/'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    root_folder = 'outputs/'
    save_model_folder = root_folder + 'models/'

    datamodule = Galaxy10_Dataset('Galaxy10.h5')
    model = AutoencoderLightning(latent_size, lr = learning_rate)

    earlystopping_tracking = 'trn_mse_loss'
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
        accelerator = None,
        max_epochs = 200, min_epochs = 5,
        default_root_dir = root_folder,
        fast_dev_run=False,
        check_val_every_n_epoch=1,
        callbacks=  [checkpoint_callback,
                                earlystop_callback,],
    )

    trainer.fit(model, datamodule = datamodule)

    encoder_model_file = save_model_folder+'encoder/'
    model_filename = f'autoencoder_{latent_size}.pt'
    if not os.path.exists(encoder_model_file):
        os.mkdir(encoder_model_file)
    encoder_model_file += model_filename

    torch.save(model.model.state_dict(), encoder_model_file)
    
    ##########################################################################
    #   LINEAR EVALUATION
    ##########################################################################

    lin_Eval = DSModelLightning(10,latent_size,True,learning_rate,encoder_model_file)

    earlystopping_tracking = 'val_loss'
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

    model_file = save_model_folder+'lin_Eval/'
    filename = f'DSModel.{latent_size}.pt'
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    model_file += filename

    torch.save(lin_Eval.model.state_dict(),model_file)

    ##########################################################################
    #   FINE TUNING 
    ##########################################################################
    
    lin_Eval = DSModelLightning(10,latent_size,False,learning_rate,encoder_model_file)
    
    earlystopping_tracking = 'val_loss'
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

    model_file = save_model_folder+'fine_tuning/'
    filename = f'DSModel.{latent_size}.pt'
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    model_file+=filename 
    torch.save(lin_Eval.model.state_dict(),model_file)    

