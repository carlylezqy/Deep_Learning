import os
import torch
import network
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import wandb
import numpy as np

import torchmetrics as tm

def train(train_loader, model, device, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item(), "train_accuracy": tm.functional.accuracy(pred, target)})

@torch.no_grad()
def test(test_loader, model, device, epoch):
    model.eval()
    test_batch_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        test_batch_loss += F.nll_loss(output, target, reduction='sum').item()
        
    test_batch_loss /= len(test_loader.dataset)
    wandb.log({"test_loss": test_batch_loss, "test_accuracy": tm.functional.accuracy(pred, target)})

def main():
    BATCH_SIZE = 64
    wandb.init(
        project="hello", name=f"MNIST_b{BATCH_SIZE}", entity="carlylezqy",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("use cudnn:", torch.backends.cudnn.benchmark)
    else:
        device = torch.device('cpu')

    dataset_part_1 = datasets.MNIST(u'/home/akiyo/nfs/dataset', train=True, download=True, transform=transform)
    dataset_part_2 = datasets.MNIST(u'/home/akiyo/nfs/dataset', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_part_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset_part_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    model = network.Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    epochs = 5

    for epoch in range(epochs):
        train(train_loader, model, device, optimizer, epoch)
        test(test_loader, model, device, epoch)
        scheduler.step()
    
    wandb.finish()
    #torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()