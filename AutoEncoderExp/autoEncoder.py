import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from resNet18 import ResNet_18
from sklearn.metrics import (accuracy_score,f1_score,auc,precision_recall_curve,roc_curve)
import torch.optim as optim 


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
        self.t_conv6 = nn.ConvTranspose2d(16, 3, 2, stride=2)
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
        
#         print(x.shape)
        x = torch.relu(self.t_conv6(x))
        x = x.view(x.size(0), 3, 128, 128)
        return x

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
#         self.cov2d  = nn.ConvTranspose2d(512, 512, (1,4), stride=2)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
#         x = self.cov2d(x)
        x = F.interpolate(x, scale_factor=4)
#         print(x.shape)
        x = self.layer4(x)
#         print(x.shape)
        x = self.layer3(x)
#         print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = x.view(x.size(0), 3, 128, 128)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self, encoder_output_num):
        super(ConvAutoencoder, self).__init__()
#         ## encoder layers ##
        self.encoder = ResNet_18(3,encoder_output_num)
        self.decoder = ResNet18Dec(z_dim = encoder_output_num)

    def forward(self, x):
        x = x.float()
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
        self.save_hyperparameters()
        self.verbose = verbose
        self.lr = lr
        self.encoder_output_num  = encoder_output_num
        self.automatic_optimization = False
        self.loss_function = nn.MSELoss()
        self.model = ConvAutoencoder(encoder_output_num)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X,y = batch
        X = X.float()
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self(X)
        loss = self.loss_function(outputs, X)
        self.manual_backward(loss)
        opt.step()
        torch.cuda.empty_cache()

        return torch.tensor(loss.item())

    def validation_step(self, batch, batch_idx):
        X,y = batch
        outputs = self(X)
        loss = self.loss_function(outputs,X)
        return torch.tensor(loss.item())
    
    def test_step(self, batch, batch_idx):
        X,y = batch
        outputs = self(X)
        loss = self.loss_function(outputs,X)
        return torch.tensor(loss.item())

    def training_epoch_end(self, outputs):
        loss = sum(item['loss'] for item in outputs) / len(outputs)
        epoch = self.current_epoch
        self.log('trn_mse_loss', loss)
        print("\nepoch_tst:Ep%d || MSE Loss:%.03f \n"%(epoch,loss))
    
    def validation_epoch_end(self, outputs):
        loss = sum(outputs) / len(outputs)
        epoch = self.current_epoch
        self.log('val_mse_loss', loss)
        print("\nepoch_val:Ep%d || MSE Loss:%.03f \n"%(epoch,loss))

    def test_epoch_end(self, outputs):
        loss = sum(outputs) / len(outputs)
        epoch = self.current_epoch
        self.log('tst_mse_loss',loss)
        print("\nepoch_tst:Ep%d || MSE Loss:%.03f \n"%(epoch,loss))

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
        # autoEncoder.load_state_dict(torch.load(autoencoder_param_dir))
        if autoencoder_param_dir:
            autoEncoder.load_state_dict(torch.load(autoencoder_param_dir))
        self.model = DSModel(autoEncoder, num_classes, encoder_output_num, linEval)


    def forward(self, batch):
        X,y = batch
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X,y = batch
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self.forward(batch)
        loss = self.loss_function(outputs, y)
        self.manual_backward(loss)
        opt.step()
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        
        return return_dict

    def validation_step(self, batch, batch_idx):
        X,y = batch
        outputs = self.forward(batch)
        loss = self.loss_function(outputs, y)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def test_step(self, batch, batch_idx):
        X,y = batch
        outputs = self(batch)
        loss = self.loss_function(outputs,y)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def training_epoch_end(self, outputs):

        sch = self.lr_schedulers()
        sch.step()

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
        self.log('trn_perf',perf_dict)
        print("\nepoch_trn:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,loss,perf_dict['Acc'],perf_dict['F1']))

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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min = 1e-08)
        return [optimizer], [scheduler]    