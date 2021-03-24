# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:27:43 2021

@author: Abdelrahman
"""
import torch
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cpu")


train_transform = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.ToTensor()])

train_set = datasets.ImageFolder(root = "dataset\\train",transform = train_transform)
train_loader = DataLoader(train_set,
                          batch_size=32,
                          shuffle=True,
                          num_workers = 0)

valid_data = datasets.ImageFolder(root = "dataset\\valid", transform = train_transfrom)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=0)

test_data = datasets.ImageFolder(root = "dataset\\test", transform = train_transfrom)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=0)

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)


print("mean:", mean.numpy()) #mean: [0.70756066 0.59119403 0.5465341 ]
print("std:", std.numpy())  #std: [0.15324508 0.16183697 0.17682996]


mean_v, std_v = get_mean_std(valid_loader)


print("mean_v:", mean_v.numpy()) #mean_v: [0.69324577 0.56011826 0.5092703 ]
print("std_v:", std_v.numpy())  #std_v: [0.13990514 0.1405701  0.15759519]


mean_t, std_t = get_mean_std(test_loader)


print("mean_t:", mean_t.numpy()) #mean_t: [0.7131744  0.550645   0.50956434]
print("std_t:", std_t.numpy())  #std_t: [0.15762942 0.16314395 0.1775014 ]
