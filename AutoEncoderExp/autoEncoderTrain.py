import torch.optim as optim 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tensorflow.keras.utils import to_categorical



# from astroNN.datasets import galaxy10
# from astroNN.datasets.galaxy10 import galaxy10cls_lookup

from resNet18 import ResNet_18

# Import data

# images, labels = galaxy10.load_data()
# labels = labels.astype(np.float32)
# labels = to_categorical(labels)
# images = images.astype(np.float32)
# images = images/255
# images = images.transpose((0,3, 1, 2))
images = torch.load('dataset_final.pt').share_memory_()
images = torch.permute(images, (0,3,1,2))

labels = torch.zeros(images.size(0)).share_memory_()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoader

# X = torch.from_numpy(images)
# y = torch.from_numpy(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.20)



datasetTensor = TensorDataset(X_train,y_train)
trainLoader = DataLoader(datasetTensor, batch_size = 32, shuffle=True)


datasetTensor = TensorDataset(X_test,y_test)
testLoader = DataLoader(datasetTensor, batch_size = 32, shuffle=True)




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
#         x = torch.relu(self.t_conv5(x))
# #         print(x.shape)
#         x = x.view(x.size(0), 3, 69, 69)
        x = torch.relu(self.t_conv6(x))
        x = x.view(x.size(0), 3, 128, 128)
#         print(x.shape)
        return x



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
#         ## encoder layers ##
        self.encoder = ResNet_18(3,256)
        self.decoder = Decoder(256)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = x.float()
        x = self.encoder(x)
#         x = x.view(2048,1,1)
#         print(x.shape)
        ## decode ##
        x = self.decoder(x)
        return x


class DSModel(nn.Module):
    def __init__(self,model,num_classes, lineEval):
        super().__init__()
        
        self.Encoder = model.encoder
        self.num_classes = num_classes
        
        if(linEval):
            for p in self.Encoder.parameters():
                p.requires_grad = False

        self.lastlayer = nn.Linear(256,self.num_classes)
        
    def forward(self,x):
        x = self.Encoder(x)
        x = self.lastlayer(x)
        
        return x


################################  TRAINING  ################################

model = ConvAutoencoder().to(device)
# criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 100
for epoch in (range(1, EPOCHS+1)):
    train_loss = 0.0
    for data in tqdm(trainLoader):
        images, _ = data
        images = images.to(device)
        images = images.float()
        outputs = model(images)
#         print(outputs.shape)
#         print(images.shape)
#         print(images.shape)
#         print(outputs.shape)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
    train_loss = train_loss/len(trainLoader)
    
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
        
                
###########################  FINE TUNNING/ LINEAR EVALUATION #########################

linEval = True


DSmodel = DSModel(model,10).to(device)
optimizer = torch.optim.Adam(DSmodel.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



for epoch in range(EPOCHS):
    for data in trainLoader:
        
        x,y = data

        
        x = x.to(device)
        y = y.to(device)
        
        outputs = DSmodel(x)
        loss = criterion(outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
    print(f'loss for this epoch {epoch + 1} is {loss.tolist()}')







############################ TESTING ################################


## with torch.no_grad():
n_correct = 0
n_samples = 0
n_class_correct = [0 for i in range(10)]
n_class_samples = [0 for i in range(10)]
for images, labels in testLoader:
    images = images.to(device)
    labels = labels.to(device)
    _, labels = torch.max(labels.data, 1)
    outputs = DSmodel(images)
    # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
    



acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network: {acc} %')
