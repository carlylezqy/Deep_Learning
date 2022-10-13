import os
import torch
import network
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

def train(train_loader, model, device, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()



        writer.add_scalars('train/epoch' + str(epoch), 
                            {'train_loss':loss.item(),
                             'train_accuracy':100. * batch_idx / len(train_loader)},
                            batch_idx)
        '''
        print('Train Epoch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss.item()))
        '''

    #if epoch % 10 == 5:
        #writer.add_images('train/', , epoch)

def test(test_loader, model, device, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    writer.add_scalars('test_total', 
                        {'test_ave_loss':test_loss,
                        'test_accuracy':100. * correct / len(test_loader.dataset)},
                        epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )
    
    #viz.line([100. * correct / len(test_loader.dataset)], [idx], win="test_loss", update="append")


def main():
    writer = SummaryWriter(comment="_LR_1.0_BATCH_64", log_dir='summary')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print("use cudnn:", torch.backends.cudnn.benchmark)
    else:
        device = torch.device('cpu')

    dataset_part_1 = datasets.MNIST(u'C:\ANN\Dataset', train=True, download=True, transform=transform)
    dataset_part_2 = datasets.MNIST(u'C:\ANN\Dataset', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_part_1, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset_part_2, batch_size=100, shuffle=True, num_workers=os.cpu_count())

    model = network.Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    epochs = 5

    writer.add_hparams(
        {'basic/lr': 1.0, 'basic/batch': 64},
        {'scheduler/step_size': 1, 'scheduler/gamma': 0.7}
    )

    for epoch in range(epochs):
        train(train_loader, model, device, optimizer, epoch, writer)
        test(test_loader, model, device, epoch, writer)
        scheduler.step()
    
    writer.add_graph(model, torch.rand(1, 1, 28, 28).to(device))
    
    writer.close()
    torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()