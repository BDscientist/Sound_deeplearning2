import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
import os
import PYCLASS2_air_Copy1
from torch.utils.data import Dataset, DataLoader


sound_train = PYCLASS2_air_Copy1.my_datset2(train = True)
                         

sound_test = PYCLASS2_air_Copy1.my_datset2(train = False)



batch_size = 25

train_loader= DataLoader(dataset=sound_train, batch_size=batch_size, shuffle=False ,num_workers=1)

test_loader = DataLoader(dataset=sound_test,  batch_size=batch_size, shuffle=False, num_workers=1)



class CNN2(nn.Module):
    def __init__(self):
        
        super(CNN2, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=16,kernel_size = 3),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2,2)
            
        )
        
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*6*1,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,3),
            nn.Softmax()
            #nn.Softmax(dim=1)
        )       
        
    def forward(self,x):
        print(x.data.shape)
        out = self.layer(x)
        out = out.view(out.shape[0],-1)
        out = self.fc_layer(out)

        return out
        
model2 = CNN2().cuda()
loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model2.parameters())
        
        
num_epochs = 300


save_path = '/home/libedev/mute/mute-hero/download/dataset/model2/'
model_path = save_path + 'model1.pkl'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
    
if os.path.isfile(model_path):
    model2.load_state_dict(torch.load(model_path))
    print("Model Loaded!")

else:
    
    for epoch in range(num_epochs):

        total_batch = len(sound_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.cuda()
            Y = batch_labels.cuda()

            pre = model2(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    if not os.path.isfile(model_path):
        print("Model Saved!")
        torch.save(model2.state_dict(), model_path)
        
        
        
        
        
        
        
        
        
   