import os
import timm
import torch
import shutil
import datatool
from tqdm import tqdm
from evaluate_tool import accuracy
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from transforms import ContrastiveLearningViewGenerator

data_root_folder = "/mnt/d/Datasets"
dataset_name = "cifar10"
batch_size = 40
workers = 8
lr = 0.0003
weight_decay = 1e-4
epochs = 300
n_views = 2
temperature=0.07
device = torch.device('cpu')
model_name = "resnet50"

def info_nce_loss(features):
    global batch_size
    global n_views
    global device
    global temperature

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = torch.nn.functional.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def train(epoch, train_loader, model, device, optimizer, scheduler, criterion):
    fp16_precision = False
    scaler = GradScaler(enabled=fp16_precision)

    mean_loss = 0

    for images in tqdm(train_loader):
        #print(images[0].shape)
        images = torch.cat(images, dim=0)
        images = images.to(device)

        with autocast(enabled=fp16_precision):
            features = model(images)

            logits, labels = info_nce_loss(features)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss += loss.item()
    
    mean_loss /= len(train_loader)
    top1 = accuracy(logits, labels)
    print(f"Epoch {epoch}: loss: {mean_loss}, top1_accuracy: {top1[0].item()}%")


class ResNetSimCLR(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(ResNetSimCLR, self).__init__()
        if model_name == "resnet18":
            self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
        if model_name == "resnet50":
            self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)

        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(dim_mlp, dim_mlp), 
            torch.nn.ReLU(), 
            self.backbone.fc
        )

    def forward(self, x):
        return self.backbone(x)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    global device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    #dataset = datatool.Dataset()
    #train_dataset = dataset.get_dataset(data_root_folder, dataset_name, train=True, n_views_transform=True)

    data_dir = "/mnt/d/Datasets/Retinal_Image_Database/DRIVE/"
    train_dataset = datatool.DRIVE_DATASET(
        data_dir, 
        "DRIVE.csv", 
        train=False, 
        transform=ContrastiveLearningViewGenerator(128), 
        single_opt=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=workers, pin_memory=True
    )

    model = ResNetSimCLR(model_name, 128).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )
    
    for epoch in range(epochs):
        train(epoch, train_loader, model, device, optimizer, scheduler, criterion)
        if(epoch >= 10):
            scheduler.step()
        
        checkpoint_name = f"checkpoint_{epoch}.pth.tar"

        save_checkpoint({
            'epoch': epochs, 'arch': model_name,
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            }, 
            is_best=False, 
            filename=os.path.join(os.path.dirname(__file__), "checkpoint", checkpoint_name)
        )
    
    

if __name__ == "__main__":
    main()