import os
import time
import torch
import datetime

import lighting_models

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.tensorboard import SummaryWriter

def main():
    seed_everything(43)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 16 if AVAIL_GPUS else 64
   
    now_time = datetime.datetime.now().strftime("%Y.%m.%d--%H:%M:%S")
    logs_root = os.path.join("/home/akiyo/logs", now_time)
    os.makedirs(logs_root, exist_ok=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    #logger = TensorBoardLogger("/home/akiyo/logs", name=f"UNet-NonePre-{BATCH_SIZE}")
    logger = TensorBoardLogger("/home/akiyo/logs", name=now_time)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/IoU",
        dirpath=logs_root,
        filename=f"checkpoint",
    )

    model = lighting_models.SegmentModule(BATCH_SIZE, logs_root)
    #checkpoint = "/home/akiyo/logs/2022-06-03_16-31-06/checkpoint.ckpt"
    checkpoint = None
    if checkpoint is not None:
        model = model.load_from_checkpoint(checkpoint, BATCH_SIZE=BATCH_SIZE, logs_root=logs_root)
    trainer = Trainer(
        amp_backend="apex",
        amp_level="O1",
        precision=16,
        accelerator="gpu",
        gpus=AVAIL_GPUS,
        max_epochs=300,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
    #trainer.test(ckpt_path='best')

if __name__ == "__main__":
    main()