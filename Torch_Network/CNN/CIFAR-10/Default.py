import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

def train(train_loader, model, device, criterion, optimizer):
    model.train()
    running_train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)

        loss = criterion(output, target)
        running_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(
                'Train Epoch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    running_train_loss / len(train_loader)
                    )
                )
            running_train_loss = 0.0

def test(test_loader, model, device, criterion):
    model.eval()
    running_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_test_loss += criterion(output, target).item()  # sum up batch loss
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    running_test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_test_loss, correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )

if __name__ == '__main__':
    dataset_name = "cifar-10-batches-py"
    image_size = [32, 32]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomResizedCrop(299, scale=(0.75, 0.75), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("use cudnn:", torch.backends.cudnn.enabled)
    else:
        device = torch.device('cpu')

    train_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/datasets', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/datasets', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

    model = models.inception_v3(aux_logits=False, init_weights=True).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    epochs = 30

    for epoch in range(epochs):
        train(train_loader, model, device, criterion, optimizer)
        test(test_loader, model, device, criterion)
        scheduler.step()

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)