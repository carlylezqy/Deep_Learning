import torch
import torchvision
import pytorch_lightning
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule, Trainer

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(aux_logits=False, init_weights=True)
        
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=0.1)
        return self.optimizer

def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    model = Model()
    
    train_dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/datasets', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=12, pin_memory=True)

    trainer = Trainer(
        accelerator="auto",
        devices=[1, 2, 3] if torch.cuda.is_available() else None,  # limiting got iPython runs
        #strategy='dp',
        strategy='ddp_find_unused_parameters_false',
        max_epochs=3,
    )

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()