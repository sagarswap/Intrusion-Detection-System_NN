import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

#device config
device=torch.device('cpu')

#hyper parameters
input_size=16
hidden_size=100
num_classes=1
num_epochs=13
batch_size=300
learning_rate=0.0002

#MNIST
class LogDatasetTrain(Dataset): 
    def __init__(self):
        xy=np.loadtxt("./data/CorrelationDataTrain.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.y=torch.FloatTensor(xy[:, -1:]) #Take last col
        self.X=torch.FloatTensor(xy[:, :-1]) #Take all but the last col
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class LogDatasetTest(Dataset): 
    def __init__(self):
        xy=np.loadtxt("./data/CorrelationDataTest.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.y=torch.FloatTensor(xy[:, -1:]) #Take last col
        self.X=torch.FloatTensor(xy[:, :-1]) #Take all but the last col
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
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
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (x_tr, y_tr) in enumerate(train_loader):
        #forward
        y_pred=model.forward(x_tr)
        loss=criterion(y_pred, y_tr)
        #print(y_pred)
        #print(y_tr)
        #print("Done")

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
        for i, data in enumerate(images):
            n_samples+=1
            y_val=model.forward(data)
            if y_val>0.5 and labels[i]==1.0:
                n_correct+=1
            elif y_val<0.5 and labels[i]==0.0:
                n_correct+=1
    acc=100.0*n_correct/n_samples
    print(f'Accuracy = {acc}')