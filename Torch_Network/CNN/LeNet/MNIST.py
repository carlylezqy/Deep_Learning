import torch
import time
import os
import network
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

def train(train_loader, model, device, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        '''
        print('Train Epoch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss.item()))
        '''
def test(test_loader, model, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print("use cudnn:", torch.backends.cudnn.benchmark)
    else:
        device = torch.device('cpu')

    dataset_part_1 = datasets.MNIST(u'/home/akiyo/nfs/dataset', train=True, download=True, transform=transform)
    dataset_part_2 = datasets.MNIST(u'/home/akiyo/nfs/dataset', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_part_1, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset_part_2, batch_size=100, shuffle=True, num_workers=os.cpu_count())

    model = network.Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    epochs = 15

    for epoch in range(epochs):
        time1 = time.time()
        train(train_loader, model, device, optimizer)
        test(test_loader, model, device)
        time2 = time.time()
        print("time in", epoch ,"epoch:", time2 - time1)
        scheduler.step()


    torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()