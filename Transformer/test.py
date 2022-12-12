import os
import glob
import torch
import random

import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from linformer import Linformer
# from vit_pytorch.efficient import ViT
from pytorch_pretrained_vit import ViT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    batch_size = 32
    epochs = 100
    lr = 3e-4
    gamma = 0.7
    seed = 42
    
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("Is CUDNN being used:", torch.backends.cudnn.enabled)
    else:
        device = torch.device('cpu')

    dataset_dir = "/home/akiyo/nfs/dataset/dogs-vs-cats-redux-kernels-edition"
    
    train_list = glob.glob(os.path.join(dataset_dir, "train", '*.jpg'))
    test_list  = glob.glob(os.path.join(dataset_dir, "test",  '*.jpg'))

    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=42)

    train_transforms = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ])

    trainval_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/nfs/dataset', train=True, download=True, transform=train_transforms)
    trainval_len, val_len = int(len(trainval_dataset) * 0.8), int(len(trainval_dataset) * 0.2)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [trainval_len, val_len])
    test_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/nfs/dataset', train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ViT('B_16_imagenet1k', pretrained=True).to(device)

    criterion = torch.nn.CrossEntropyLoss()                    # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # scheduler

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        progress = tqdm(train_loader)

        for data, label in progress:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            progress.set_postfix(acc=acc.item(), loss=loss.item())

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

if __name__ == '__main__':
    seed_everything(42)
    main()