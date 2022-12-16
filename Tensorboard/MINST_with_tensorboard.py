import torch
import torchvision
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

def train(model, trainloader, criterion, optimizer, writer, epoch, device):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/akiyo/logs"),
    ) as profiler:
        model.train()
        for batch_idx, (images, target) in enumerate(trainloader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            grid = torchvision.utils.make_grid(images)
            writer.add_image('images', grid, 0)

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                
                writer.add_graph(model, images)
                writer.add_scalar('Train/Loss', loss.item(), epoch)
                profiler.step()
        
def main():
    writer = SummaryWriter(log_dir="/home/akiyo/logs")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = datasets.FashionMNIST("/home/akiyo/datasets", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(2):
        train(model, trainloader, criterion, optimizer, writer, epoch, device)
    
    writer.close()

if __name__ == "__main__":
    main()