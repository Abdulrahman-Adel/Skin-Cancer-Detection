# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:50:02 2021

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
from torchvision import transforms, datasets, models
import torch
from torch import nn


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

device = torch.device('cpu')


######################################## 
########## Loading Test Data ###########
########################################


test_transfrom = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])


test_data = datasets.ImageFolder(root = "dataset\\test", transform = test_transfrom)
testloader = torch.utils.data.DataLoader(test_data,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=0)

paths = [i[0] for i in test_data.samples]

######################################## 
########## Testing the model ###########
########################################


def Test_model(model, SK = True):
    with torch.no_grad():
        model.eval()
        predictions = []
        for data_t, target_t in (testloader):
            if SK:
                target_t = np.array([1 if i == 2 else 0 for i in target_t.numpy()])
                target_t = torch.tensor(target_t.astype(np.longlong))
            else:
                target_t = np.array([1 if i == 0 else 0 for i in target_t.numpy()])
                target_t = torch.tensor(target_t.astype(np.longlong))
                
                
                     
            data_t, target_t = data_t.to(device), target_t.to(device) 
             
            
            outputs_t = model(data_t)
            
            _,pred_t = torch.max(outputs_t, dim=1)
            predictions.append(pred_t)
        
        predictions = torch.cat(predictions).cpu().numpy()
        
        return predictions
    
    
######################################## 
######### Loading saved models #########
########################################  


model = models.resnet50(pretrained = True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

PATH_SK = "resnet_SK.pt"
PATH_MM = "resnet_MM.pt"


model.load_state_dict(torch.load(PATH_SK))
predictions_SK = Test_model(model)


model.load_state_dict(torch.load(PATH_MM))
predictions_MM = Test_model(model, SK = False)

######################################## 
########## Preparing csv file ##########
########################################

submission = pd.DataFrame({"id":paths,
                           "task_1":list(predictions_SK),
                           "task_2":list(predictions_MM)})


submission.to_csv("submission.csv", index = False)







        