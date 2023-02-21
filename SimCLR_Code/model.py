import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning.losses import NTXentLoss
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from sklearn.metrics import (accuracy_score,f1_score,auc,precision_recall_curve,roc_curve)

transforms_dict = {'hflip':transforms.RandomHorizontalFlip(),
                   'crop+resize': transforms.RandomResizedCrop(size=96),
                   'colorjitter': transforms.RandomApply([transforms.ColorJitter(brightness=0.5,
                                                                                 contrast=0.5,
                                                                                 saturation=0.5,
                                                                                 hue=0.1)], p=0.8),
                   'gray': transforms.RandomGrayscale(p=0.2),
                   'blur': transforms.GaussianBlur(kernel_size=9)}

t2np = lambda t: t.detach().cpu().numpy()

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )

class SimCLR(nn.Module):
    def __init__(self, image_channels, encoder_output_num):
        super().__init__()
        self.encoder = ResNet_18(image_channels, encoder_output_num)
        self.projection = nn.Sequential(
            nn.Linear(encoder_output_num, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.projection(x) 
        return x

class SimCLR_container(LightningModule):
    def __init__(self, image_channels, encoder_output_num, verbose=False, lr = 0.001, num_out_dim = 10, 
                 transforms_list = ['hflip','crop+resize','colorjitter','gray','blur'], temperature = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.verbose = verbose
        self.lr = lr
        self.num_classes = num_out_dim
        self.image_channels = image_channels
        self.encoder_out_dim = encoder_output_num
        self.automatic_optimization = False
        self.loss_function = NTXentLoss(temperature=temperature)

        self.contrast_transforms = transforms.Compose([transforms_dict[transform] for transform in transforms_list] +
                                                      [transforms.Normalize((0.5,),(0.5,))])


        self.model = SimCLR(self.image_channels,self.encoder_out_dim)

    def forward(self, batch):
        x, y = batch
        x1 = self.contrast_transforms(x)
        x2 = self.contrast_transforms(x)
        x1 = self.model(x1)
        x2 = self.model(x2)
        return torch.cat((x1,x2))

    def training_step(self, batch, batch_idx):
        X, y = batch
        opt = self.optimizers()
        opt.zero_grad()
        embeddings = self(batch)
        indices = torch.arange(0,embeddings.size(0)//2)
        labels = torch.cat((indices,indices))
        loss = self.loss_function(embeddings,labels)
        self.manual_backward(loss)
        opt.step()
        self.log("trn_ntxent_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        embeddings = self(batch)
        indices = torch.arange(0,embeddings.size(0)//2)
        labels = torch.cat((indices,indices))
        loss = self.loss_function(embeddings,labels)
        self.log('val_ntxent_loss',loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        embeddings = self(batch)
        indices = torch.arange(0,embeddings.size(0)//2)
        labels = torch.cat((indices,indices))
        loss = self.loss_function(embeddings,labels)
        self.log('test_ntxent_loss',loss)
        return loss

    def training_epoch_end(self, outputs):
        loss = sum(item['loss'] for item in outputs) / len(outputs)
        epoch = self.current_epoch
        print("\nepoch_trn:Ep%d || NTXent Loss:%.03f \n"%(epoch,loss.item()))

    def validation_epoch_end(self, outputs):
        loss = sum(outputs) / len(outputs)
        epoch = self.current_epoch
        print("\nepoch_val:Ep%d || NTXent Loss:%.03f \n"%(epoch,loss.item()))

    def test_epoch_end(self, outputs):
        loss = sum(outputs) / len(outputs)
        epoch = self.current_epoch
        print("\nepoch_tst:Ep%d || NTXent Loss:%.03f \n"%(epoch,loss.item()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class DSModel(nn.Module):
    def __init__(self,simCLR,num_classes,linEval):
        super().__init__()
        simCLR.projection = Identity()
        self.simCLREncoder = simCLR
        self.linEval = linEval
 
        if self.linEval:
            for p in self.simCLREncoder.parameters():
                p.requires_grad = False

        self.lastlayer = nn.Linear(1024,num_classes)
        
    def forward(self,x):
        x = self.simCLREncoder(x)
        x = self.lastlayer(x)
        return x
    
class LightningDSModel(LightningModule):
    def __init__(self,image_channels,encoder_output_num,encoder_param_dir,num_classes,linEval,lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        self.linEval = linEval
        self.loss_function = nn.CrossEntropyLoss()
        self.automatic_optimization = False

        encoder = SimCLR(image_channels,encoder_output_num)
        encoder.load_state_dict(torch.load(encoder_param_dir))
        self.model = DSModel(encoder,self.num_classes,self.linEval)
    
    def forward(self, data):
        x,y = data
        return self.model(x)
    
    def training_step(self,data,batch_idx):
        x,y = data
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self(data)
        loss = self.loss_function(outputs,y)
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss',loss)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def validation_step(self,data,batch_idx):
        x,y = data
        outputs = self(data)
        loss = self.loss_function(outputs,y)
        self.log('val_loss',loss)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def test_step(self,data,batch_idx):
        x,y = data
        outputs = self(data)
        loss = self.loss_function(outputs,y)
        self.log('test_loss',loss)
        return_dict = {'loss':loss,'y_out':t2np(outputs),'y':t2np(y)}
        return return_dict

    def training_epoch_end(self, outputs):
        epoch = self.current_epoch
        y_out = np.concatenate([x['y_out'] for x in outputs])
        y = np.concatenate([x['y'] for x in outputs])
        #losses = np.concatenate([x['loss'].cpu() for x in outputs])
        #loss = sum(losses) / len(losses)
        y_pred = y_out.argmax(axis=1)
        y_true = y.argmax(axis=1)
        perf_dict = {}
        perf_dict['F1']  = f1_score(y_true,y_pred,average='macro')
        perf_dict['Acc']  = accuracy_score(y_true,y_pred)
        #perf_dict['loss'] = loss
        self.log('trn_perf',perf_dict)
        print("\nepoch_trn:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,0,perf_dict['Acc'],perf_dict['F1']))

    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch
        y_out = np.concatenate([x['y_out'] for x in outputs])
        y = np.concatenate([x['y'] for x in outputs])
        #losses = np.concatenate([x['loss'].cpu() for x in outputs])
        #loss = sum(losses) / len(losses)
        y_pred = y_out.argmax(axis=1)
        y_true = y.argmax(axis=1)
        perf_dict = {}
        perf_dict['F1']  = f1_score(y_true,y_pred,average='macro')
        perf_dict['Acc']  = accuracy_score(y_true,y_pred)
        #perf_dict['loss'] = loss
        self.log('val_perf',perf_dict)
        print("\nepoch_val:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,0,perf_dict['Acc'],perf_dict['F1']))

    def test_epoch_end(self, outputs):
        epoch = self.current_epoch
        y_out = np.concatenate([x['y_out'] for x in outputs])
        y = np.concatenate([x['y'] for x in outputs])
        #losses = np.concatenate([x['loss'].cpu() for x in outputs])
        #loss = sum(losses) / len(losses)
        y_pred = y_out.argmax(axis=1)
        y_true = y.argmax(axis=1)
        perf_dict = {}
        perf_dict['F1']  = f1_score(y_true,y_pred,average='macro')
        perf_dict['Acc']  = accuracy_score(y_true,y_pred)
        #perf_dict['loss'] = loss
        self.log('test_perf',perf_dict)
        print("\nepoch_tst:Ep%d || Loss:%.03f Accuracy:%.03f F1:%.03f\n"%(epoch,0,perf_dict['Acc'],perf_dict['F1']))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer

