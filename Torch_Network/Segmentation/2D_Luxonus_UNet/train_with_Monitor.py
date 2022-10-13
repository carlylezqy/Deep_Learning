import torch
import os, time
import DataTools
import network
from PIL import Image

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import image_transforms

def train_model(model, train_loader, optimizer):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()

    for image, mask in train_loader:
        image, mask = image.to(device), mask.to(device)
        optimizer.zero_grad()

        masks_pred = model(image)
        loss = criterion(masks_pred, mask)

        loss.backward()
        optimizer.step()

        print(loss.item())


def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for images, index in test_loader:
            images = images.to(device)
            output = model(images)

            output = output.cpu().detach().numpy()

            for i in output:
                opt = i[0]
                opt = (opt - opt.min()) / (opt.max() - opt.min())
                opt[opt >= 0.1] = 1
                opt[opt < 0.1] = 0
                im = Image.fromarray(opt*255)
                im = im.convert("L")
                os.makedirs("output/" + index[0][0], exist_ok=True)
                im.save(u"output/" + index[0][0] + "/"+ index[1][0] + ".jpeg")
        
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print("use cudnn:", torch.backends.cudnn.benchmark)
    else:
        device = torch.device('cpu')

    batch_size = 3

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
   
    #luxonus_train = DataTools.LUXONUS_DATASET(u"C:\ANN\dataset\Luxonus_Data_HDF5", "Luxonus_Data_HDF5.csv", train=True, batch_size=30)
    #luxonus_test = DataTools.LUXONUS_DATASET(u"C:\ANN\dataset\Luxonus_Data_HDF5", "Luxonus_Data_HDF5_test.csv", train=False, batch_size=10)
    drive_train = DataTools.DRIVE_DATASET("/mnt/d/Datasets/Retinal_Image_Database/DRIVE", "DRIVE.csv", train=True, transform=transform, batch_size=40)
    drive_test = DataTools.DRIVE_DATASET("/mnt/d/Datasets/Retinal_Image_Database/DRIVE", "DRIVE.csv", train=True, transform=transform, batch_size=40)
    train_loader_luxonus = DataLoader(drive_train, batch_size=40, num_workers=os.cpu_count())
    test_loader_luxonus = DataLoader(drive_test, batch_size=1, num_workers=os.cpu_count())

    model = network.UNet(n_channels=1, n_classes=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)

    epochs = 10

    for epoch in range(epochs):
        train_model(model, train_loader_luxonus, optimizer)
        torch.cuda.empty_cache()
        scheduler.step()
    test_model(model, test_loader_luxonus)
    


