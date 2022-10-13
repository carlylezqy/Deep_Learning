from tkinter import Y
import torch
import time
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.loggers import TensorBoardLogger
import lighting_unet
from pytorch_lightning.callbacks import ModelCheckpoint



DATASETS_PATH = "/mnt/d/Datasets/Retinal_Image_Database/DRIVE"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 16 if AVAIL_GPUS else 64

torch.backends.cudnn.enabled = True
model_backbone = "vgg13_bn"

checkpoint =  "/home/akiyo/code/Deep_Learning/Project/Segmentation/UNet/input_checkpoint/LightlyModel-1645602031.1788187-v1.ckpt"
model = lighting_unet.UNet(DATASETS_PATH, model_backbone, BATCH_SIZE, in_channels=3, checkpoint=checkpoint)

logger = TensorBoardLogger("TensorBoardLog", name=f"U-Net_{model_backbone}")
checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="/home/akiyo/code/Deep_Learning/Project/Segmentation/UNet/checkpoint",
    filename=f"{model_backbone}-{time.strftime('%Y-%m-%d_%H:%M:%S')}",
)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=300,
    logger=logger,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback]
)

trainer.fit(model)
trainer.test(ckpt_path='best')