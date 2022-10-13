import os
import random
import time

import numpy as np
from PIL import Image

import torch
from torchmetrics import Accuracy
from torchvision import transforms
from pytorch_lightning import LightningModule

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from model.unet import unet
from model.unet import unet2
#from model import model_bkp as model
from model.unet import resnet34_unet
from model.unet import xception_unet
from model import utils
from model import deeplab

from dataset.voc2012 import VOC2012 
from dataset.oxford_pet_dataset import Oxford_Pet as dataset
from dataset.penn_fudan import Penn_Fudan 
#Penn_Fudan, VOC2012, Oxford_Pet

from model.deeplab._deeplab import ASPP

class SegmentModule(LightningModule):
    def __init__(self, BATCH_SIZE, logs_root, checkpoint=None):
        super(SegmentModule, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WORKER = 4
        
        self.BATCH_SIZE = BATCH_SIZE
        self.logs_root = logs_root
        self.img_opt_path = os.path.join(self.logs_root, "output")

        self.height, self.width = 512, 512

        self.albumentation = A.Compose([
            A.Resize(self.height, self.width, always_apply=True),
            # A.HorizontalFlip(p=0.3),
            # A.CenterCrop(height=self.height, width=self.width, p=0.1),
            # A.Blur(blur_limit=15, p=0.1),
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.num_classes = dataset.get_num_classes()

        self.model = unet.UNet(in_channels=3, n_class=self.num_classes)#; utils.init_weights(self.model, init_type='kaiming')

        
        self.accuracy = Accuracy().to(device)
        #self.criterion = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE).to(device)
        #self.criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True).to(device)
        #self.criterion = smp.losses.JaccardLoss(smp.losses.MULTICLASS_MODE).to(device)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        self.i = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.007, momentum=0.9, weight_decay=5e-4, nesterov=False)
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
        mask = self.model(image)
        return mask

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = dataset(
                albumentation=self.albumentation, transformer=self.transform,
                train=True
            )
            
            self.val_dataset = dataset(
                albumentation=A.Resize(self.height, self.width, always_apply=True), transformer=self.transform,
                train=False
            )

        if stage == "test" or stage is None:
            pass #self.test_dataset = datatool.DRIVE_DATASET(self.data_dir, "DRIVE.csv", (self.height, self.width), train=False, transform=None)
    
    #dataloader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=True, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.WORKER, shuffle=False, pin_memory=False, persistent_workers=True)

    def test_dataloader(self):
        #return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=self.WORKER, shuffle=False, pin_memory=False, persistent_workers=True)
        pass

    #training/val
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        #decode = lambda x: x * 255
        #decode = lambda x: general_dataset.GeneralDataset.decode(x)
        
        y_preds, ys = outputs[0]["y_pred"], outputs[0]["y"]
        i = random.randint(0, ys.shape[0]-1)
        #print(y_preds[i].shape, ys[i].shape)
        combine = np.hstack((y_preds[i], ys[i]))
        #print(combine.shape, ys[i].max(), ys[i].min())

        combine = combine[0] if combine.ndim == 3 else combine
        combine = self.train_dataset.decoder(combine)

        os.makedirs(self.img_opt_path, exist_ok=True)
        Image.fromarray(combine).save(os.path.join(self.img_opt_path, f"y_pred_{self.i}.png"))
        self.i += 1

        return self.shared_epoch_end(outputs, "valid")

    #shared-step
    @torch.no_grad()
    def evaluation(self, pred_mask, mask):
        dict = {}
        
        #prob_mask = logits_mask.sigmoid()
        #pred_mask = (torch.max(prob_mask, dim=1).indices).long()

        #print(pred_mask.shape, mask.shape)
        tp, fp, fn, tn = smp.metrics.functional.get_stats(pred_mask, mask.long(), mode="multiclass", num_classes=self.num_classes)
        #tp, fp, fn, tn = smp.metrics.functional.get_stats(pred_mask[:, 0], mask.long(), mode="binary", threshold=self.threshold)
        tp, fp, fn, tn = tp.long(), fp.long(), fn.long(), tn.long()

        accuracy = torch.mean(smp.metrics.functional.accuracy(tp, fp, fn, tn))
        sensitivity = torch.mean(smp.metrics.functional.sensitivity(tp, fp, fn, tn))
        specificity = torch.mean(smp.metrics.functional.specificity(tp, fp, fn, tn))
        iou_score = torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"))

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
        #loss = self.criterion(logits_mask, mask)
        result = self.evaluation(pred_mask, mask)

        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/IoU', result["iou_score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/Accuracy', result["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "iou_score": result["iou_score"],
            "y_pred": numpy_pred_mask,
            "y": numpy_mask,
        }

    def shared_epoch_end(self, outputs, stage):
        pass

if __name__ == "__main__":
    pass