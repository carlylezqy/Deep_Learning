import os
import cv2
import torch
import datatool
import network
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


dataset_name = "cifar-10-batches-py"
image_size = [32, 32]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 5

train_dataset = datatool.CIFAR_DATASET(dataset_name, ipt_batch_size=5, train=True, transform=transform)
test_dataset = datatool.CIFAR_DATASET(dataset_name, train=False, transform=transform)

for i in range(0, len(train_dataset), 1000):
    print(i, train_dataset[i][0].shape, train_dataset[i][1])

for i in range(0, len(test_dataset), 1000):
    print(i, test_dataset[i][0].shape, test_dataset[i][1])
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)

'''
for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape)
''' 