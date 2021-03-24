# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:11:42 2021

@author: Abdelrahman
"""

import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models
from Model import Net



mean = np.array([0.70756066,0.59119403,0.5465341 ])
std = np.array([0.15324508,0.16183697,0.17682996])

mean_v = np.array([0.69324577,0.56011826,0.5092703 ])
std_v = np.array([0.13990514,0.1405701,0.15759519])

#'cuda:0' if torch.cuda.is_available() else
device = torch.device('cpu')


######################################## 
## Loading Train and validation Data ###
########################################


train_transfrom = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])

valid_transfrom = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(torch.Tensor(mean_v), torch.Tensor(std_v))])

train_data = datasets.ImageFolder(root = "dataset\\train", transform = train_transfrom)
trainloader = torch.utils.data.DataLoader(train_data,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=0)

valid_data = datasets.ImageFolder(root = "dataset\\valid", transform = valid_transfrom)
validloader = torch.utils.data.DataLoader(valid_data,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=0)


#################################################
## Defining model, optimizer and loss function ##
#################################################

model = models.resnet50(pretrained = True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)

#model = Net()

model.to(device)

criterion = nn.CrossEntropyLoss()
#criterion = nn.HingeEmbeddingLoss()

optimizer = optim.Adagrad(model.parameters(),lr = 0.001)



step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)


########################################
############ Training model ############
########################################


def train_model(model, criterion, optimizer, scheduler, SK = True, n_epochs=5):
    
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainloader)
    
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(trainloader):
            if SK:
                target_ = np.array([1 if i == 2 else 0 for i in target_.numpy()])
                target_ = torch.tensor(target_.astype(np.longlong))
            else:
                target_ = np.array([1 if i == 0 else 0 for i in target_.numpy()])
                target_ = torch.tensor(target_.astype(np.longlong))
             
            data_, target_ = data_.to(device), target_.to(device)    
            optimizer.zero_grad()
        
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
        
            #scheduler.step()
        
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        
        batch_loss = 0
        total_v=0
        correct_v=0
        
        with torch.no_grad():
            model.eval()
            for data_v, target_v in (validloader):
                if SK:
                    target_v = np.array([1 if i == 2 else 0 for i in target_v.numpy()])
                    target_v = torch.tensor(target_v.astype(np.longlong))
                else:
                    target_v = np.array([1 if i == 0 else 0 for i in target_v.numpy()])
                    target_v = torch.tensor(target_v.astype(np.longlong))
                     
                data_v, target_v = data_v.to(device), target_v.to(device) 
                
                outputs_v = model(data_v)
                loss_v = criterion(outputs_v, target_v)
                batch_loss += loss_v.item()
                _,pred_v = torch.max(outputs_v, dim=1)
                correct_v += torch.sum(pred_v==target_v).item()
                total_v += target_v.size(0)
            val_acc.append(100 * correct_v/total_v)
            val_loss.append(batch_loss/len(validloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_v/total_v):.4f}\n')

        
            if network_learned:
                valid_loss_min = batch_loss
                if SK:
                    torch.save(model.state_dict(), 'resnet_SK.pt')
                else:
                    torch.save(model.state_dict(), 'resnet_MM.pt')
                print('Improvement-Detected, save-model') 
        model.train()
        
    return train_loss, val_loss   
        
##################################################
## seborrheic_keratosis vs. rest classification ##  
##################################################
        
train_loss, val_loss = train_model(model,
                                       criterion,
                                       optimizer,
                                       step_lr_scheduler,
                                       SK = True,
                                       n_epochs=2) 



########################################
#### Plotting Train-Validation Loss ####
########################################


fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Loss")
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')


##################################################
######## melanoma vs. rest classification ########  
##################################################

train_loss, val_loss = train_model(model,
                                       criterion,
                                       optimizer,
                                       step_lr_scheduler,
                                       SK = False,
                                       n_epochs=5)

        


