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
input_size=76
hidden_size=100
num_classes=6
num_epochs=20
batch_size=100
learning_rate=0.001

#MNIST


#train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#examples=iter(train_loader)
#samples, labels=next(examples)
#print(samples.shape, labels.shape)

class LogDataset(Dataset):
    def __init__(self):
        xy=pandas.read_csv('./data/Wednesday_wh.csv')
        #y=torch.from_numpy(xy[:, -1]) #Take last col
        #x=torch.from_numpy(xy[:, :-1]) #Take all but the last col
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

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
    
#model=NeuralNet(input_size, hidden_size, num_classes)
dataset=LogDataset()
first_data=dataset[0]
features, labels=first_data
print(features, labels)
#loss and optimizer
#criterion=nn.CrossEntropyLoss()
#optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
#n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images=images.reshape(-1, 28*28).to(device)
        labels=labels.to(device)

        #forward
        outputs=model(images)
        loss=criterion(outputs, labels)

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
        images=images.reshape(-1, 28*28).to(device)
        labels=labels.to(device)
        outputs = model(images)

        _, predicictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicictions==labels).sum().item()
    acc=100.0*n_correct/n_samples
    print(f'Accuracy = {acc}')