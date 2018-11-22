from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import resnet

plt.ion()   # interactive mode

#data loading

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'images'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = resnet.resnet18(pretrained=True)
model.load_state_dict(torch.load('./saved_model.pth'))
model.eval()

print(len(dataloaders['train']))

inputs = dataLoader['train'][0]
#inputs = inputs.to(device)
outputs = model.forward(inputs).detach()
outputs = np.array(outputs)
_,a,b,c = outputs.shape

out_all = np.empty((0,a,b,c),dtype=np.float64)

for inputs, labels in dataloaders['train']:
	inputs = inputs.to(device)
	labels = labels.to(device)

	outputs = model.forward(inputs).detach()
	outputs = np.array(outputs)
	out_all = np.append(out_all,outputs,axis=0)

print(out_all.shape)

