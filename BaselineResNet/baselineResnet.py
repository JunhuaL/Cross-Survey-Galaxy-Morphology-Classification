import torch.optim as optim 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

import h5py
from ResNet import ResNet_18


print("loading data")

with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes
print("loading complete")

print("creating categorical labels")
# To convert to desirable type
labels = np.eye(10)[labels]
labels = labels.astype(np.float16,copy=False)
images = images.astype(np.float16,copy=False)
images /= 255
images = images.transpose((0,3, 1, 2))
print("converted labels")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("splitting data and loading into tensors")

images = torch.from_numpy(images).share_memory_()
labels = torch.from_numpy(labels).share_memory_()

X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size=0.2)

print("loaded to tensor")
print("loading training set dataloader")
train_set = TensorDataset(X_train,y_train)
trainLoader = DataLoader(train_set, batch_size = 32, shuffle=True)
print("complete loading training set dataloader")

print("loading test set dataloader")
test_set = TensorDataset(X_test,y_test)
testLoader = DataLoader(test_set, batch_size = 32, shuffle=True)
print("complete loading test set dataloader")

learning_rate = 0.001
EPOCHS = 20
scheduler_patience = 5

net = ResNet_18(3,10).to(device)
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
            images = images.float()

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
        X = X.float()
        y = y.float().to(device)
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



