import torch
import torchvision
from torchmetrics import Accuracy
from torchvision import transforms
from pytorch_lightning import LightningModule

class LightingNet(LightningModule):
    def __init__(self, data_dir, BATCH_SIZE, hidden_size=64, learning_rate=2e-4):
        super(LightingNet, self).__init__()
        self.WORKER = 8

        self.data_dir = data_dir
        self.BATCH_SIZE = BATCH_SIZE
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output
    

    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.BATCH_SIZE, num_workers=self.WORKER)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.BATCH_SIZE, num_workers=self.WORKER)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.BATCH_SIZE)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.nll_loss(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.nll_loss(self(x), y)
        preds = torch.argmax(self(x), dim=1)
        self.accuracy(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=1.0)