import time
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as models

labels_name = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

def train(epoch, model, dataloader, device, criterion, optimizer):
    model.train()
    train_mean_loss = 0

    correct_pred = {classname: 0 for classname in labels_name}
    total_pred = {classname: 0 for classname in labels_name}

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            predictions = model(images)
            loss = criterion(predictions, labels)
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_mean_loss += loss.item()
        _, predictions = torch.max(predictions, dim=1)
        accuracy = metrics(predictions, labels)

        second = int(time.strftime("%S", time.localtime()))
        if second % 10 == 1:
            print(f"train_loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}")

        optimizer.step()
    
    train_mean_loss /= len(dataloader)
    print(f"[epoch {i}] loss: {train_mean_loss}")

@torch.no_grad()
def validate(epoch, model, dataloader, device, criterion):
    model.eval()
    
    val_mean_loss = 0

    correct_pred = {classname: 0 for classname in labels_name}
    total_pred = {classname: 0 for classname in labels_name}

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            predictions = model(images)
            loss = criterion(predictions, labels)
        
        val_mean_loss += loss.item()
        
        _, predictions = torch.max(predictions, 1)
        accuracy = metrics(predictions, labels)

        second = int(time.strftime("%S", time.localtime()))
        if second == 1:
            print(f"val_loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}")
        
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[labels_name[label]] += 1
            total_pred[labels_name[label]] += 1

    
    val_mean_loss /= len(dataloader)
    print(f"[epoch {i}] loss: {val_mean_loss}")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:8s} is {accuracy:.3f} %')

if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()
    ])

    trainval_dataset = torchvision.datasets.FashionMNIST('/home/akiyo/datasets', train=True, download=True, transform=transformer)
    test_dataset = torchvision.datasets.FashionMNIST('/home/akiyo/datasets', train=False, download=True, transform=transformer)

    train_size = int(len(trainval_dataset) * 0.8)
    val_size = int(len(trainval_dataset) - train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    BATCH_SIZE = 128

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=len(labels_name), bias=True)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = lambda predictions, labels: predictions.eq(labels).sum().detach().cpu().numpy() / labels.shape[0]

    EPOCH = 1
    AMP_ENABLED = True
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    for i in range(EPOCH):
        train(i, model, train_dataloader, device, criterion, optimizer)
        validate(i, model, val_dataloader, device, criterion)

    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    torch.save(model, f"{current_time}.pth")