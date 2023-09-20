import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas

#device config
device=torch.device('cpu')

#hyper parameters
input_size=23
hidden_size=50
num_classes=6
num_epochs=10
batch_size=100
learning_rate=0.001

#MNIST
class LogDatasetTrain(Dataset): 
    def __init__(self):
        xy=np.loadtxt("./data/train_data.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.y=torch.from_numpy(xy[:, -6]) #Take last col
        self.x=torch.from_numpy(xy[:, :-6]) #Take all but the last col
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class LogDatasetTest(Dataset): 
    def __init__(self):
        xy=np.loadtxt("./data/test_data.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.y=torch.from_numpy(xy[:, -6]) #Take last col
        self.x=torch.from_numpy(xy[:, :-6]) #Take all but the last col
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
train_data=LogDatasetTrain()
test_data=LogDatasetTest()
train_loader=torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out
    
model=NeuralNet(input_size, hidden_size, num_classes)
#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (x_tr, y_tr) in enumerate(train_loader):

        #forward
        y_pred=model(x_tr)
        #print(y_pred)
        #print(y_tr)
        #print(type(y_pred[0]))
        #print(type(y_tr[0]))
        y_tr=y_tr.type(torch.LongTensor)
        loss=criterion(y_pred, y_tr)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#test
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        print(predicictions)
        print(labels)
        _, predicictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicictions==labels).sum().item()
    acc=100.0*n_correct/n_samples
    print(f'Accuracy = {acc}')