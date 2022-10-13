import os
import random
import cv2
import time
import glob
from pyparsing import Combine
from sklearn.metrics import jaccard_score
from sklearn.utils import shuffle
import torch
import torchvision
from torchmetrics import Accuracy
from torchvision import transforms
from pytorch_lightning import LightningModule
from models.layers import unetConv2, unetUp, unetUp_origin
from models.init_weights import init_weights_mtd
from pytorch_toolbelt import losses as L

import pandas as pd
from torch.optim import lr_scheduler

import datatool
from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp
import lighting_unet_org

import evaluate_tool
import torchmetrics

from sklearn.metrics import roc_auc_score
from pytorch_lightning import seed_everything

t = time.localtime()
#seed_everything(t.tm_min)
seed_everything(42)

#warnings.filterwarnings('ignore')
from albumentations.pytorch import ToTensorV2
import albumentations as A

import network

class UNet(LightningModule):
    def __init__(self, data_dir, model_name, BATCH_SIZE, in_channels=3, n_classes=1, checkpoint=None):
        super(UNet, self).__init__()
        self.model_name = model_name
        self.WORKER = 8
        
        self.data_dir = data_dir
        self.BATCH_SIZE = BATCH_SIZE
        self.accuracy = Accuracy()
        
        self.height, self.width = 640, 640

        """
        A.OneOf([
                A.RandomSizedCrop(min_max_height=(200, self.height), height=self.height, width=self.width, p=0.5),
                A.PadIfNeeded(min_height=self.height, min_width=self.width, p=0.5)
            ], p=1),
        """

        self.transform = A.Compose([
            A.Resize(self.height, self.width, always_apply=True),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.5),
            #A.RandomRotate90(p=0.5),
            #A.MotionBlur(always_apply=False, p=0.3),
            #A.GaussianBlur(always_apply=False, p=0.3),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=45, interpolation=2, border_mode=1, p=0.3),
            #A.CLAHE(p=0.5),
            #A.RandomBrightnessContrast(p=0.5),
            #A.RandomGamma(p=0.5),
            #A.ElasticTransform(p=0.5),
            #A.HueSaturationValue(p=0.5),
            #A.RandomCrop(400, 400, p=0.5),
            #A.Resize(self.height, self.width, always_apply=True),
            A.RandomGamma(p=1, gamma_limit=(100, 200)),
            A.CLAHE(p=1, clip_limit=(1, 6)),
        ])

        original_height, original_width = self.height, self.width

        #self.transform = None

        self.model = network.UNet(n_channels=3, n_classes=1)
        
        #self.model.encoder.load_state_dict(torch.load(checkpoint))
        params = smp.encoders.get_preprocessing_params(self.model_name, "imagenet")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        #self.criterion2 = smp.losses.SoftCrossEntropyLoss()
        #self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.sce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
        
        #self.criterion = torch.nn.BCEWithLogitsLoss()

        self.threshold = 0.5
        #self.metrics = JaccardIndex(num_classes=2)
        self.i = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
        ) 
        """
        return [optimizer, ]#, [scheduler, ]

    def forward(self, image):
        inputs = (image - self.mean) / self.std
        mask = self.model(inputs)
        return mask

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainval_dataset = datatool.DRIVE_DATASET(self.data_dir, "DRIVE.csv", (self.height, self.width), train=True, transform=self.transform)
            
            total = len(self.trainval_dataset)
            train_num = int(total * 0.8)
            #self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.trainval_dataset, [train_num, total - train_num])

            self.train_dataset, self.val_dataset = self.trainval_dataset, self.trainval_dataset
            self.val_dataset.transform = None
            
            val_dataset_dir = "/mnt/d/Datasets/Retinal_Image_Database/VEVIO"
            dataset3_dir = "/mnt/d/Datasets/Retinal_Image_Database/STARE_Segmentation"
            #self.val_dataset.train_image_list = glob.glob(os.path.join(val_dataset_dir, 'mosaics', '*.png'))
            #self.val_dataset.train_groud_list = glob.glob(os.path.join(val_dataset_dir, 'mosaics_manual_01_bw', '*.png'))
            self.val_dataset.train_image_list = glob.glob(os.path.join(dataset3_dir, 'Image', '*.ppm'))
            self.val_dataset.train_groud_list = glob.glob(os.path.join(dataset3_dir, 'Mask', '*.ppm'))

        if stage == "test" or stage is None:
            self.test_dataset = datatool.DRIVE_DATASET(self.data_dir, "DRIVE.csv", (self.height, self.width), train=False, transform=None)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=self.WORKER, shuffle=False)

    def shared_step(self, batch, stage):
        image, mask = batch["image"].permute(0, 3, 1, 2), batch["mask"].permute(0, 3, 1, 2)

        mask = mask / 255.0
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        #mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and self.criterion param `from_logits` is set to True
        loss = self.criterion(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.threshold).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.functional.get_stats(pred_mask.long(), mask.long(), mode="binary", threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        numpy_pred_mask = logits_mask.sigmoid().cpu().detach().numpy()
        numpy_mask = mask.long().cpu().numpy()
        
        mean_auc_score = []
        mean_dis = []
        for i, j in zip(numpy_mask, numpy_pred_mask):
            try:
                mean_auc_score.append(roc_auc_score(i.flatten(), j.flatten()))
                mean_dis.append(evaluate_tool.dice(i, j))
            except ValueError:
                pass

        auc_score = np.mean(mean_auc_score)

        tp, fp, fn, tn = tp.long(), fp.long(), fn.long(), tn.long()

        accuracy = torch.mean(smp.metrics.functional.accuracy(tp, fp, fn, tn))
        sensitivity = torch.mean(smp.metrics.functional.sensitivity(tp, fp, fn, tn))
        specificity = torch.mean(smp.metrics.functional.specificity(tp, fp, fn, tn))
        dice_dis = np.mean(mean_dis)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_iou', iou_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_auc', auc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_Accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_Dice', dice_dis, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "iou_score": iou_score,
            "y_pred": np.uint8(pred_mask.cpu() * 255),
            "y": np.uint8(mask.cpu() * 255),
            "auc_score": mean_auc_score,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        """
        metrics = {
            f"{stage}_loss": torch.cat([x["loss"] for x in outputs]),
            f"{stage}_iou_score": torch.cat([x["iou_score"] for x in outputs]),
        }
        
        self.log_dict(metrics, prog_bar=True)
        """
        pass

    def training_step(self, batch, batch_idx):

        """
        x, y = batch
        logits_mask = self(x)
        loss = self.criterion(logits_mask, y)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.threshold).float()

        restore_y = (y.clone().sigmoid() > self.threshold).float()

        jaccard_value = evaluate_tool.jaccard(restore_y, pred_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_jaccard', jaccard_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        print("loss: ", loss.item())
        return {'loss': loss, 'pred_mask': pred_mask, 'jaccard': jaccard_value}
        """
        return self.shared_step(batch, "train")            


    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")
        #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_jaccard = torch.stack([x['jaccard'] for x in outputs]).mean()
        #print(f"\n{avg_loss}, {avg_jaccard}")

    def validation_step(self, batch, batch_idx):
        """
        x, y = batch
        logits_mask = self(x)
        loss = self.criterion(logits_mask, y)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.threshold).float()

        restore_y = (y.clone().sigmoid() > self.threshold).float()
        jaccard_value = evaluate_tool.jaccard(restore_y, pred_mask)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_jaccard', jaccard_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'pred_mask': pred_mask, 'jaccard': jaccard_value}
        """
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_jaccard = torch.stack([x['jaccard'] for x in outputs]).mean()
        i = random.randint(0, len(outputs[0]["y_pred"])-1)
        y_pred, y = outputs[0]["y_pred"][i, 0], outputs[0]["y"][i, 0]

        combine = np.hstack((y_pred, y))
        Image.fromarray(combine).save(f"/home/akiyo/code/Deep_Learning/Project/Segmentation/UNet/test/y_pred_{self.i}.png")
        self.i += 1

        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        x = batch["image"].permute(0, 3, 1, 2)
        logits_mask = self(x)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.threshold).float()

        image = Image.fromarray(np.uint8(pred_mask[0, 0].cpu() * 255))
        #image = image.resize((565, 584))
        image.save(f"/home/akiyo/code/Deep_Learning/Project/Segmentation/UNet/test/{batch_idx + 1}.png")

if __name__ == "__main__":
    module = UNet("/mnt/d/Datasets/Retinal_Image_Database/DRIVE/", 64)
    module.prepare_data()