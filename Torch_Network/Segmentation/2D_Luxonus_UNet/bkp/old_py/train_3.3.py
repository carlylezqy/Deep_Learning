import torch
import os, time
import DataTools
import network
from unet import UNet
from PIL import Image

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import image_transforms

def train_model(model, train_loader, optimizer, scheduler=None):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0

    for batch_idx, (image, mark) in enumerate(train_loader):
        image, mark = image.to(device), mark.to(device)
        optimizer.zero_grad()
        
        masks_pred = model(image)
        loss = criterion(masks_pred, mark)
        
        loss.backward()
        optimizer.step()

        '''
        ipt = image[0][0].cpu().numpy()
        opt = mark[0][0].cpu().numpy()
        
        print(np.min(ipt), np.max(ipt))
        print(np.min(opt), np.max(opt))
        print("+++")

        #output = (output - output.min()) / (output.max() - output.min())
        im = Image.fromarray(np.uint8(output))
        im = im.convert("RGB")
        im.save(str(time.time()) + "XXXXX.jpeg")
        '''
        print(loss.item())

def test_model(model, test_loader):
    model.eval()
    for images in test_loader:
        for image in images:
            image = image.to(device)
            #print(image.shape)
            output = model(image)
            '''
            ipt = image[0][0].cpu().numpy()
            print(np.min(ipt), np.max(ipt))
            print(np.min(output), np.max(output))
            #output = (output > 0.5).astype(np.uint8)
            '''
            output = output[0][0].cpu().detach().numpy()
            output = (output - output.min()) / (output.max() - output.min())

            im = Image.fromarray(output*255)
            im = im.convert("L")
            im.save(str(time.time()) + ".jpeg")
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 3

    transform = image_transforms.transform
    '''
    train_set_drive = DataTools.DRIVE_DATASET("./dataset/DRIVE", "DRIVE.csv", batch_size, train=True, transform=transform)
    test_set_drive = DataTools.DRIVE_DATASET("./dataset/DRIVE", "DRIVE.csv", batch_size, train=False, transform=transform)
    train_set_chasedb1 = DataTools.DRIVE_DATASET("./dataset/CHASEDB1", "CHASEDB1.csv", batch_size, train=True, transform=transform)
    train_loader_drive = DataLoader(train_set_drive, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    train_loader_chasedb1 = DataLoader(train_set_chasedb1, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    '''
   
    dataset_luxonus = DataTools.LUXONUS_DATASET("./dataset/luxonus_data", "luxonus_data.csv", 30)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_luxonus, [20, 10])
    train_loader_luxonus = DataLoader(train_dataset, batch_size=10, num_workers=os.cpu_count())
    test_loader_luxonus = DataLoader(test_dataset, batch_size=1, num_workers=os.cpu_count())

    #model = u_net.UNet(1).to(device)
    #model = network.UNet(n_channels=3, n_classes=1).to(device)
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    epochs = 10
    '''
    for i in range(epochs):
        train_model(model, train_loader_drive, optimizer)#, scheduler)
    
    for i in range(epochs):
        train_model(model, train_loader_chasedb1, optimizer)
    '''
    for i in range(epochs):
        train_model(model, train_loader_luxonus, optimizer, scheduler)
        test_model(model, test_loader_luxonus)
        scheduler.step()

    
    #test_model(model, test_loader_drive)
    


