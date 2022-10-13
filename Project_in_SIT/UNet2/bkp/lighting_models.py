import os
import random
import cv2
import time
import glob
from cv2 import threshold
from pyparsing import Combine
from sklearn.metrics import jaccard_score
from sklearn.utils import shuffle
import torch
import torchvision
from torchmetrics.functional import accuracy
from torchmetrics.functional import auc
from torchvision import transforms
from pytorch_lightning import LightningModule

import datetime
import pandas as pd
from torch.optim import lr_scheduler

from PIL import Image
import numpy as np

import torchmetrics
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score
from pytorch_lightning import seed_everything

t = time.localtime()
#seed_everything(t.tm_min)
seed_everything(42)

import general_dataset
#warnings.filterwarnings('ignore')
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn import metrics

class SegmentModule(LightningModule):
    def __init__(self, BATCH_SIZE, logs_root, checkpoint=None):
        super(SegmentModule, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WORKER = 8
        
        self.BATCH_SIZE = BATCH_SIZE
        self.logs_root = logs_root
        self.img_opt_path = os.path.join(self.logs_root, "output")
        os.makedirs(self.img_opt_path, exist_ok=True)

        self.height, self.width = 256, 256

        self.albumentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomGamma(p=0.2, gamma_limit=(80, 150)),
                A.CLAHE(p=0.5),#, clip_limit=(10, 50), tile_grid_size=(5, 5)),
                A.RandomBrightnessContrast(p=0.2, brightness_limit=(0.2, 0.3), contrast_limit=(0.1, 0.4), always_apply=False),
            ]),
            A.ElasticTransform(p=0.3, alpha=1, sigma=23, alpha_affine=40),
            A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), rotate_limit=0, scale_limit=(-0.2, 0.2), p=0.5),
            A.Resize(self.height, self.width, always_apply=True),
        ])

        self.albumentation = A.Compose([A.Resize(self.height, self.width, always_apply=True)])

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
        ])
        self.num_classes = 2

        self.model = smp.Unet(
            encoder_name="xception",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,        # model output channels (number of classes in your dataset)
            #decoder_attention_type="scse",
        )

        #self.accuracy = accuracy().to(device)
        self.criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True).to(device)
        
        self.auroc = torchmetrics.AUROC(num_classes=self.num_classes)

        self.threshold = 0.5
        self.i = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, weight_decay=5e-4, nesterov=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后,T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
        ) 
        """
        return [optimizer, ]#, [scheduler, ]

    def forward(self, image):
        mask = self.model.forward(image)
        return mask

    def setup(self, stage=None):
        data_path = "/home/akiyo/datasets/Retinal_Image_Database/DRIVE/training/images"
        label_path = "/home/akiyo/datasets/Retinal_Image_Database/DRIVE/training/1st_manual"

        data_path2 = "/home/akiyo/datasets/Retinal_Image_Database/CHASEDB1/training/images"
        label_path2 = "/home/akiyo/datasets/Retinal_Image_Database/CHASEDB1/training/mask"

        data_path3 = "/home/akiyo/datasets/Retinal_Image_Database/RITE/training/images"
        label_path3 = "/home/akiyo/datasets/Retinal_Image_Database/RITE/training/vessel"

        data_path4 = "/home/akiyo/datasets/Retinal_Image_Database/RITE/test/images"
        label_path4 = "/home/akiyo/datasets/Retinal_Image_Database/RITE/test/vessel"

        if stage == "fit" or stage is None:
            # self.train_dataset = general_dataset.GeneralDataset(
            #     image_dir=data_path, masks_dir=label_path, 
            #     image_extension="_training.tif", mask_extension="_manual1.gif", 
            #     albumentation=self.albumentation, transformer=self.transform,
            #     special_rendering=general_dataset.GeneralDataset.special_rendering
            # )

            # self.train_dataset.append_extra_list(
            #     image_dir=data_path4, masks_dir=label_path4, 
            #     image_extension=".tif", mask_extension=".png",
            # )

            # self.train_dataset.append_extra_list(
            #     image_dir=data_path3, masks_dir=label_path3, 
            #     image_extension=".tif", mask_extension=".png",
            # )

            # self.val_dataset = general_dataset.GeneralDataset(
            #     image_dir=data_path2, masks_dir=label_path2, 
            #     image_extension=".jpg", mask_extension="_1stHO.png",
            #     albumentation=A.Resize(self.height, self.width, always_apply=True), transformer=self.transform,
            #     special_rendering=general_dataset.GeneralDataset.special_rendering
            # )
            
            def endcoder(name, image, mask):
                mask[mask == 255] = 1
                mask = mask[0]
                return image, mask

            data_path = "/home/akiyo/datasets/Membrane/train/image"
            label_path = "/home/akiyo/datasets/Membrane/train/label"
            self.trainval_dataset = general_dataset.GeneralDataset(
                image_dir=data_path, masks_dir=label_path, 
                image_extension=".png", mask_extension=".png", 
                albumentation=self.albumentation, transformer=self.transform,
                #special_rendering=GeneralDataset.special_rendering
                special_rendering=endcoder
            )
            
            total = len(self.trainval_dataset)
            train_num = int(total * 0.8)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.trainval_dataset, [train_num, total - train_num])
        if stage == "test" or stage is None:
            pass #self.test_dataset = datatool.DRIVE_DATASET(self.data_dir, "DRIVE.csv", (self.height, self.width), train=False, transform=None)
    
    #dataloader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=False, pin_memory=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=self.WORKER, shuffle=False)

    #training/val
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        def decode(combine):
            return combine * 255
            #y_cmp = self.trainval_dataset.decode_segmap(np.asarray(combine, dtype=np.uint8))
            #y_cmp = np.uint8(y_cmp * 255)
            #return y_cmp

        y_preds, ys = outputs[0]["y_pred"], outputs[0]["y"]
        i = random.randint(0, ys.shape[0]-1)
        #print(y_preds[i].shape, ys[i].shape)
        combine = np.hstack((y_preds[i], ys[i]))
        #print(combine.shape)
        combine = combine[0] if combine.ndim == 3 else combine
        combine = decode(combine)
        #combine = general_dataset.GeneralDataset.decode(combine)
        Image.fromarray(combine).save(os.path.join(self.img_opt_path, f"y_pred_{self.i}.png"))
        self.i += 1
        torch.cuda.empty_cache()
        return self.shared_epoch_end(outputs, "valid")

    #shared-step
    def evaluation(self, pred_masks, masks):
        dict = {}
    
        tp, fp, fn, tn = smp.metrics.functional.get_stats(pred_masks, masks.long(), mode="multiclass", num_classes=self.num_classes)
        tp, fp, fn, tn = tp.long(), fp.long(), fn.long(), tn.long()

        accuracy = torch.mean(smp.metrics.functional.accuracy(tp, fp, fn, tn))
        sensitivity = torch.mean(smp.metrics.functional.sensitivity(tp, fp, fn, tn))
        specificity = torch.mean(smp.metrics.functional.specificity(tp, fp, fn, tn))
        iou_score = torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"))
        
        #print(pred_masks.max(), pred_masks.min(), masks.max(), masks.min())
        dict = {
            "iou_score": iou_score,
            "accuracy": accuracy,
        }

        return dict

    def shared_step(self, batch, stage):
        image, mask = batch[0], batch[1]#.unsqueeze(1)
        #image, mask = batch[0], batch[1].permute(0, 3, 1, 2)
        image, mask = image.float(), mask.long()
        #mask[mask != 1] = 0
        #mask[mask > 0] = 1
        assert image.ndim == 4 #and mask.ndim == 4
        #assert mask.min() >= 0 and mask.max() <= 1.0

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"Wrong shape {h} x {w}"

        #train
        logits_mask = self.forward(image)#['out']
        loss = self.criterion(logits_mask, mask)

        #print(logits_mask.shape)
        if logits_mask.shape[1] != 1:
            pred_mask = torch.argmax(logits_mask, dim=1)  #the activation of segmentation head is softmax
        else:
            pred_mask = torch.round(logits_mask).int()   #the activation of segmentation head is sigmoid

        numpy_mask = mask.cpu().detach().numpy().astype(np.uint8)
        numpy_pred_mask = pred_mask.cpu().detach().numpy().astype(np.uint8)

        #loss
        #print(logits_mask.shape, mask.shape)
        result = self.evaluation(pred_mask, mask)
        auc = self.auroc(logits_mask, mask)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_IoU', result["iou_score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_Accuracy', result["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_AUC', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            "loss": loss,
            "iou_score": result["iou_score"],
            "y_pred": numpy_pred_mask,
            "y": numpy_mask,
        }

    def shared_epoch_end(self, outputs, stage):
        pass

if __name__ == "__main__":
    module = SegmentModule("/mnt/d/Datasets/Retinal_Image_Database/DRIVE/", 64)
    module.prepare_data()