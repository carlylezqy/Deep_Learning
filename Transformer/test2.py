import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

# from model import ViT
from pytorch_pretrained_vit import ViT

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 32
EPOCHS = 50
BATCHES_PER_EPOCH = 100
PRINT = True
PRINT_GRAPH = True
PRINT_CM = True


# measures accuracy of predictions at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader, num_classes=10):
    correct = 0
    cm = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i > BATCHES_PER_EPOCH:
                break
            y_ = model(x.to(device))
            y_ = torch.argmax(y_, dim=1)

            correct += (y_ == y.to(device))

    print(correct / BATCHES_PER_EPOCH)
    return (correct / BATCHES_PER_EPOCH).item() * 100


# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    accuracy_per_epoch_train = []
    accuracy_per_epoch_val = []

    for epoch in range(EPOCHS):
        model.train()
        progress = tqdm(train_loader)
        for i, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)
            
            if i > BATCHES_PER_EPOCH:
                break

            y_ = model(x)
            loss = loss_function(y_, y)

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item())

            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()


        model.eval()

        accuracy_per_epoch_train.append(accuracy(model, train_loader))
        accuracy_per_epoch_val.append(accuracy(model, validation_loader))

        if PRINT:
            print(f"Test Accuracy for epoch{epoch} is: {np.round(accuracy_per_epoch_val[-1], 2)}")
            print(f"Train Accuracy for epoch{epoch} is: {np.round(accuracy_per_epoch_train[-1], 2)}")

        if PRINT_GRAPH:
            plt.figure(figsize=(10, 10), dpi=100)
            plt.plot(range(0, epoch + 1), accuracy_per_epoch_train,
                     color='b', marker='o', linestyle='dashed', label='Training')
            plt.plot(range(0, epoch + 1), accuracy_per_epoch_val,
                     color='r', marker='o', linestyle='dashed', label='Validation')
            plt.legend()
            plt.title("Graph of accuracy over time")
            plt.xlabel("epoch #")
            plt.ylabel("accuracy %")
            if epoch < 20:
                plt.xticks(range(0, epoch + 1))
            plt.ylim(0, 100)
            plt.show()
        

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("Is CUDNN being used:", torch.backends.cudnn.enabled)
    else:
        device = torch.device('cpu')


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
    # train_dataset = torchvision.datasets.MNIST(
    #     '/home/akiyo/nfs/dataset',
    #     train=True,
    #     download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             (0.1307,), (0.3081,))
    #     ]))

    # validation_dataset = torchvision.datasets.MNIST(
    #     '/home/akiyo/nfs/dataset',
    #     train=False,
    #     download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             (0.1307,), (0.3081,))
    #     ]))


    trainval_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/nfs/dataset', train=True, download=True, transform=train_transforms)
    trainval_len, val_len = int(len(trainval_dataset) * 0.8), int(len(trainval_dataset) * 0.2)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [trainval_len, val_len])
    test_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/nfs/dataset', train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True)

    validation_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False)

    # x, y = train_dataset[0]
    # # print(x.size)
    # plt.imshow(x)
    # plt.show()
    # print(y)

    #model = ViT(in_channels=3, num_classes=10, device=device).to(device)
    model = ViT('B_16_imagenet1k', pretrained=True).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, loss_function, optimizer, train_loader, validation_loader)