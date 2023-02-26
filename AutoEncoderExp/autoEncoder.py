import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from resNet18 import ResNet_18
from sklearn.metrics import (accuracy_score,f1_score,auc,precision_recall_curve,roc_curve)

t2np = lambda t: t.detach().cpu().numpy()




class Decoder(nn.Module):
    def __init__(self, inputDim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(inputDim,512)
        self.t_conv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(16, 3, 6, stride=1)
    def forward(self, z):
        x = self.linear(z)
#         print(x.shape)
        x = x.view(z.size(0),512,1,1)
#         print(x.shape)
        x = F.interpolate(x,scale_factor=4)
#         print(x.shape)
        x = torch.relu(self.t_conv1(x))
#         print(x.shape)
        x = torch.relu(self.t_conv2(x))
#         print(x.shape)
        x = torch.relu(self.t_conv3(x))
#         print(x.shape)
        x = torch.relu(self.t_conv4(x))
#         print(x.shape)
        x = torch.relu(self.t_conv5(x))
#         print(x.shape)
        x = x.view(x.size(0), 3, 69, 69)
#         print(x.shape)
        return x



class ConvAutoencoder(nn.Module):
    def __init__(self, encoder_output_num):
        super(ConvAutoencoder, self).__init__()
#         ## encoder layers ##
        self.encoder = ResNet_18(3,encoder_output_num)
        self.decoder = Decoder(encoder_output_num)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.encoder(x)
#         x = x.view(2048,1,1)
#         print(x.shape)
        ## decode ##
        x = self.decoder(x)
        return x


class AutoencoderLightning(LightningModule):
    def __init__(self, encoder_output_num, verbose=False, lr = 0.001):
        super().__init__()
        # TODO ?
        self.save_hyperparameters()
        # TODO ?
        self.verbose = verbose
        self.lr = lr
        self.encoder_output_num  = encoder_output_num
        # TODO ?
        self.automatic_optimization = False
        self.loss_function = nn.MSELoss()
        self.model = ConvAutoencoder(encoder_output_num)

    def forward(self, batch):
        X,y = batch
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        X,y = batch
        outputs = self.forward(X)
        loss = self.loss_function(y, outputs)
        return loss

    def training_epoch_end(self, outputs):
        loss = sum(item['loss'] for item in outputs) / len(outputs)
        epoch = self.current_epoch
        self.log('MSE_loss', loss)
        print("\nepoch_tst:Ep%d || MSE Loss:%.03f \n"%(epoch,loss.item()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer




class DSModel(nn.Module):
    def __init__(self,model,num_classes, encoder_output_num, linEval):
        super().__init__()
        
        self.Encoder = model.encoder
        self.num_classes = num_classes
        
        if(linEval):
            for p in self.Encoder.parameters():
                p.requires_grad = False

        self.lastlayer = nn.Linear(encoder_output_num,self.num_classes)
        
    def forward(self,x):
        x = self.Encoder(x)
        x = self.lastlayer(x)
        
        return x


class DSModelLightning(LightningModule):
    def __init__(self, num_classes, encoder_output_num, linEval, lr, autoencoder_param_dir):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        self.linEval = linEval
        self.loss_function = nn.CrossEntropyLoss()
        self.automatic_optimization = False     

        autoEncoder = ConvAutoencoder(encoder_output_num)
        autoEncoder.load_state_dict(torch.load(autoencoder_param_dir))
        self.model = DSModel(autoEncoder, num_classes, encoder_output_num, linEval)


    def forward(self, batch):
        X,y = batch
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X,y = batch
        outputs = self.forward(batch)
        loss = self.loss_function(outputs, y)
        return loss 

    def validation_step(self, batch, batch_idx):
        X,y = batch
        outputs = self.forward(batch)
        loss = self.loss_function(outputs, y)
        return_dict = {'loss':loss, 'y_out':t2np(outputs), 'y':t2np}
        return return_dict

    def test_step(self, batch, batch_idx):
        X,y = batch
        outputs = self(batch)
        loss = self.loss_function(outputs,y)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch
        y_out = np.concatenate([x['y_out'] for x in outputs])
        y = np.concatenate([x['y'] for x in outputs])
        losses = np.array([x['loss'].cpu().item() for x in outputs])
        loss = sum(losses) / len(losses)
        y_pred = y_out.argmax(axis=1)
        y_true = y.argmax(axis=1)
        perf_dict = {}
        perf_dict['F1']  = f1_score(y_true,y_pred,average='macro')
        perf_dict['Acc']  = accuracy_score(y_true,y_pred)
        perf_dict['loss'] = loss
        self.log('val_perf',perf_dict)
        self.log('val_loss',loss)
        print("\nepoch_val:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,loss,perf_dict['Acc'],perf_dict['F1']))

    def test_epoch_end(self, outputs):
        epoch = self.current_epoch
        y_out = np.concatenate([x['y_out'] for x in outputs])
        y = np.concatenate([x['y'] for x in outputs])
        losses = np.array([x['loss'].cpu().item() for x in outputs])
        loss = sum(losses) / len(losses)
        y_pred = y_out.argmax(axis=1)
        y_true = y.argmax(axis=1)
        perf_dict = {}
        perf_dict['F1']  = f1_score(y_true,y_pred,average='macro')
        perf_dict['Acc']  = accuracy_score(y_true,y_pred)
        perf_dict['loss'] = loss
        self.log('test_perf',perf_dict)
        print("\nepoch_tst:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,loss,perf_dict['Acc'],perf_dict['F1']))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer    