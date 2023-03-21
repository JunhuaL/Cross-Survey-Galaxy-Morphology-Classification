import torch.optim as optim 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import h5py
from ResNet import ResNet_18
import sys
from sklearn.metrics import (balanced_accuracy_score,accuracy_score,f1_score,auc,precision_recall_curve,roc_curve)

class GaussianNoise:
    def __init__(self,mean=0.,std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self,tensor):
        return tensor.cuda() + torch.cuda.FloatTensor(tensor.size()).normal_() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean},std={self.std})'

transforms_dict = {'crop+resize': transforms.RandomApply([transforms.RandomResizedCrop(size=256)],p=0.5),
                   'colorjitter': transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.1),
                   'gray': transforms.RandomApply([transforms.RandomGrayscale(p=0.2)],p=0.5),
                   'blur': transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)],p=0.5),
                   'rotation': transforms.RandomRotation(degrees=(0,360)),
                   'gauss_noise': transforms.RandomApply([GaussianNoise(mean=0, std=0.05)],p=0.5)}

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


print("loading data")

dataset = sys.argv[1]
use_balance = sys.argv[2]
balance_count = sys.argv[3]
if use_balance.lower() == 'true':
    use_balance = True
else:
    use_balance = False
if balance_count.strip() == 'None':
    balance_count = None
else:
    balance_count = int(balance_count)

with h5py.File(dataset, 'r') as F:
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

X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.2)

if use_balance:
    X_train, y_train = getBalanceDataset(X_train, y_train, balance_count, ['colorjitter','rotation','gauss_noise'])
    train_images = t2np(X_train)
    train_labels = t2np(y_train)
    train_labels = train_labels.argmax(axis=1)
    labels = np.unique(train_labels)
    X_train ,y_train,X_valid,y_valid = [],[],[],[]
    for label in labels:
        data_split = train_test_split(train_images[train_labels==label],train_labels[train_labels==label],test_size=0.1)
        X_train.append(data_split[0])
        X_valid.append(data_split[1])
        y_train.append(np.eye(10)[data_split[2]])
        y_valid.append(np.eye(10)[data_split[3]])
    X_train = torch.from_numpy(np.concatenate(X_train)).share_memory_()
    y_train = torch.from_numpy(np.concatenate(y_train)).share_memory_()
    X_valid = torch.from_numpy(np.concatenate(X_valid)).share_memory_()
    y_valid = torch.from_numpy(np.concatenate(y_valid)).share_memory_()
else:
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.1)

print("loaded to tensor")
print("loading training set dataloader")
train_set = TensorDataset(X_train,y_train)
trainLoader = DataLoader(train_set, batch_size = 32, shuffle=True)
print("complete loading training set dataloader")

print("loading validation set dataloader")
valid_set = TensorDataset(X_valid,y_valid)
validLoader = DataLoader(valid_set, batch_size = 32, shuffle=True)
print("complete loading validation set dataloader")

print("loading test set dataloader")
test_set = TensorDataset(X_test,y_test)
testLoader = DataLoader(test_set, batch_size = 32, shuffle=True)
print("complete loading test set dataloader")

learning_rate = 0.001
EPOCHS = 100
scheduler_patience = 5

transforms_list = ['gauss_noise','crop+resize','crop+resize']
data_aug = transforms.Compose([transforms_dict[transform] for transform in transforms_list])
net = ResNet_18(3,10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min = 1e-08)

best_f1 = 0
best_epoch = 0
best_model = net.state_dict()

def test():
    global best_f1
    global best_epoch
    trueLabels = []
    predictLabels = []

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validLoader:
            images, labels = data
            labels = labels.float().to(device)
            images = images.float().to(device)
            images = torch.cat((images,data_aug(images.clone().detach().to(device))))
            labels = torch.cat((labels,labels.clone().detach().to(device)))
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            _, labels = torch.max(labels.data, 1)

            trueLabels.append(t2np(labels))
            predictLabels.append(t2np(predicted))
            correct += (predicted == labels).sum().item()

    y_true = np.concatenate(trueLabels)
    y_pred = np.concatenate(predictLabels)
    perf_dict = {}
    perf_dict['Micro_F1'] = f1_score(y_true,y_pred,average='micro')
    perf_dict['Macro_F1']  = f1_score(y_true,y_pred,average='macro')
    perf_dict['Micro_Acc']  = accuracy_score(y_true,y_pred)
    perf_dict['Macro_Acc'] = balanced_accuracy_score(y_true,y_pred)
    if perf_dict['Macro_F1'] > best_f1:
        best_f1 = perf_dict['Macro_F1']
        best_epoch = epoch
    print(f"val_perf:{epoch} || Micro F1: {perf_dict['Micro_F1']}, Macro F1: {perf_dict['Macro_F1']}, Micro Acc: {perf_dict['Micro_Acc']}, Macro Acc: {perf_dict['Macro_Acc']}")
    #print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

train_acc = []
train_loss = []

for epoch in range(EPOCHS):
    losses = []
    train_running_correct = 0
    trueLabels = []
    predictLabels = []
    for data in trainLoader:
        X,y = data
        X = X.float().to(device)
        y = y.float().to(device)
        X = torch.cat((X,data_aug(X.clone().detach().to(device))))
        y = torch.cat((y,y.clone().detach().to(device)))
        net.zero_grad()
        output = net(X.to(device))
        _, preds = torch.max(output.data, 1)
        _, labels = torch.max(y,1)
        train_running_correct += (preds == labels).sum().item()
        trueLabels.append(t2np(labels))
        predictLabels.append(t2np(preds))

        loss = criterion(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    y_true = np.concatenate(trueLabels)
    y_pred = np.concatenate(predictLabels)
    perf_dict = {}
    perf_dict['Micro_F1'] = f1_score(y_true,y_pred,average='micro')
    perf_dict['Macro_F1']  = f1_score(y_true,y_pred,average='macro')
    perf_dict['Micro_Acc']  = accuracy_score(y_true,y_pred)
    perf_dict['Macro_Acc'] = balanced_accuracy_score(y_true,y_pred)
    print(f"trn_perf:{epoch} || Micro F1: {perf_dict['Micro_F1']}, Macro F1: {perf_dict['Macro_F1']}, Micro Acc: {perf_dict['Micro_Acc']}, Macro Acc: {perf_dict['Macro_Acc']}")
    
    mean_loss = sum(losses)/len(losses)
    print(f'loss for this epoch {epoch} is {mean_loss}')
    test()
    if best_epoch == epoch:
        best_model = net.state_dict()
    lr_scheduler.step()

print(f"best model at epoch {best_epoch}")
net = ResNet_18(3,10).to(device)
net.load_state_dict(best_model)
with torch.no_grad():
    trueLabels = []
    predictLabels = []
    correct = 0
    for data in testLoader:
        images, labels = data
        labels = labels.float().to(device)
        images = images.float().to(device)
        images = torch.cat((images,data_aug(images.clone().detach().to(device))))
        labels = torch.cat((labels,labels.clone().detach().to(device)))
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.to(device)
        _, labels = torch.max(labels.data, 1)

        trueLabels.append(t2np(labels))
        predictLabels.append(t2np(predicted))
        correct += (predicted == labels).sum().item()
    
y_true = np.concatenate(trueLabels)
y_pred = np.concatenate(predictLabels)
perf_dict = {}
perf_dict['Micro_F1'] = f1_score(y_true,y_pred,average='micro')
perf_dict['Macro_F1']  = f1_score(y_true,y_pred,average='macro')
perf_dict['Micro_Acc']  = accuracy_score(y_true,y_pred)
perf_dict['Macro_Acc'] = balanced_accuracy_score(y_true,y_pred)
print(f"tst_perf:{epoch} || Micro F1: {perf_dict['Micro_F1']}, Macro F1: {perf_dict['Macro_F1']}, Micro Acc: {perf_dict['Micro_Acc']}, Macro Acc: {perf_dict['Macro_Acc']}")



