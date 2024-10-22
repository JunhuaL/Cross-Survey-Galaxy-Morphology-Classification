import torch.optim as optim 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

import h5py
from VGG08 import VGG08




with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes


# To convert to desirable type
labels = np.eye(10)[labels]
labels = labels.astype(np.float32)
images = images.astype(np.float32)
images = images/255
images = images.transpose((0,3, 1, 2))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



X = torch.from_numpy(images)
y = torch.from_numpy(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


datasetTensor = TensorDataset(X_train,y_train)
trainLoader = DataLoader(datasetTensor, batch_size = 32, shuffle=True)


datasetTensor = TensorDataset(X_test,y_test)
testLoader = DataLoader(datasetTensor, batch_size = 32, shuffle=True)

learning_rate = 0.001
EPOCHS = 20
scheduler_patience = 5

net = VGG08(3,10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)


def test():

    trueLabels = []
    predictLabels = []

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testLoader:
            images, labels = data

            # calculate outputs by running images through the network
            outputs = net(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            _, labels = torch.max(labels.data, 1)

            trueLabels.extend(labels.tolist())
            predictLabels.extend(predicted.tolist())
    #         for i in range(len(predicted)):
    #             if predicted[i] == labels[i]:
    #                 correct += 1

            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

train_acc = []
train_loss = []

for epoch in range(EPOCHS):
    losses = []
    train_running_correct = 0
    for data in trainLoader:
        X,y = data
        y = y.to(device)
        net.zero_grad()
        # print(type(X))
        # print(X.shape)
        output = net(X.to(device))
        # print(output)
        # print(y)

#         _, preds = torch.max(output.data, 1)
#         train_running_correct += (preds == y).sum().item()

        _, preds = torch.max(output.data, 1)
        _, trueLabels = torch.max(y,1)
        train_running_correct += (preds == trueLabels).sum().item()


        loss = criterion(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    mean_loss = sum(losses)/len(losses)
    print(f'loss for this epoch {epoch + 1} is {mean_loss}')
    
    test()
    epoch_acc = 100. * (train_running_correct / len(trainLoader.dataset))
    print(f'train accuracy for this epoch {epoch + 1} is {epoch_acc}')
    train_acc.append(epoch_acc)
    train_loss.append(mean_loss)
    lr_scheduler.step(mean_loss)



