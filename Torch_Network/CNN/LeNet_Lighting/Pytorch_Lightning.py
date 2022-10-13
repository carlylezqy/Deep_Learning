import torch
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST

import lighting_network

DATASETS_PATH = "/mnt/d/Datasets"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

model = lighting_network.LightingNet(DATASETS_PATH, BATCH_SIZE)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=30,
)

trainer.fit(model)
#trainer.test()