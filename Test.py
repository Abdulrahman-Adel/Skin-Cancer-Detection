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


mean = np.array([0.7131744,0.550645,0.50956434])
std = np.array([0.15762942,0.16314395,0.1775014 ])

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
        for data_t, _ in (testloader):
            
                
            data_t = data_t.to(device)
             
            
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
                           "task_1":list(predictions_MM),
                           "task_2":list(predictions_SK)})


submission.to_csv("submission.csv", index = False)







        