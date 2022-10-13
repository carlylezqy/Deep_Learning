import os
import glob
import torch
import pandas as pd
from PIL import Image
from libtiff import TIFF
import numpy as np
import random

from albumentations.pytorch import ToTensorV2
import albumentations as A

class DRIVE_DATASET(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, csv_file, size, train=True, transform=None, single_opt=False):
        super().__init__()
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.train = train
        self.single_opt = single_opt
        self.height, self.width = size

        dataset_dir = "/mnt/d/Datasets/Retinal_Image_Database/DRIVE"
        self.train_image_list = glob.glob(os.path.join(dataset_dir, 'training', 'images', '*.tif'))
        self.train_groud_list = glob.glob(os.path.join(dataset_dir, 'training', '1st_manual', '*.gif'))
        self.test_image_list = glob.glob(os.path.join(dataset_dir, 'test', 'images', '*.tif'))

        dataset2_dir = "/mnt/d/Datasets/Retinal_Image_Database/CHASEDB1"
        self.train_image_list += glob.glob(os.path.join(dataset2_dir, 'training', 'images', '*.jpg'))
        self.train_groud_list += glob.glob(os.path.join(dataset2_dir, 'training', 'mask', '*_1stHO.png'))

        dataset4_dir = "/mnt/d/Datasets/Retinal_Image_Database/RITE"
        self.train_image_list += glob.glob(os.path.join(dataset4_dir, 'training', 'images', '*.tif'))
        self.train_groud_list += glob.glob(os.path.join(dataset4_dir, 'training', 'vessel', '*.png'))
        self.train_image_list += glob.glob(os.path.join(dataset4_dir, 'test', 'images', '*.tif'))
        self.train_groud_list += glob.glob(os.path.join(dataset4_dir, 'test', 'vessel', '*.png'))

        dataset_argment = "/mnt/d/Datasets/Retinal_Image_Database/DataArgument"
        self.train_image_list += glob.glob(os.path.join(dataset_argment, 'training', 'images', '*.png'))
        self.train_groud_list += glob.glob(os.path.join(dataset_argment, 'training', 'vessel', '*.png'))

        c = list(zip(self.train_image_list, self.train_groud_list))
        random.shuffle(c)
        self.train_image_list, self.train_groud_list = zip(*c)

        if single_opt:
            self.image_list = self.train_image_list + self.test_image_list

    def __len__(self):
        if self.single_opt:
            return len(self.image_list)
        elif self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)

    def __getitem__(self, index):
        image, mask = None, None
        if self.single_opt:
            with Image.open(self.image_list[index]) as img:
                image = img.convert('RGB').resize((self.height, self.width), resample=Image.NEAREST)
                image = np.asarray(image, dtype=np.uint8)

        else:
            if self.train:
                with Image.open(self.train_image_list[index]) as img:
                    image = img.convert('RGB')
                    if self.transform is not None:
                        image = image.resize((self.height, self.width), resample=Image.NEAREST)
                    image = np.asarray(image, dtype=np.uint8)

                with Image.open(self.train_groud_list[index]) as grd:
                    mask = grd.convert('L')
                    if self.transform is not None:
                        mask = mask.resize((self.height, self.width), resample=Image.BICUBIC)
                    mask = np.asarray(mask, dtype=np.uint8)[:, :, np.newaxis]

            else:
                with Image.open(self.test_image_list[index]) as img:
                    image = img.convert('RGB')
                    if self.transform is not None:
                        image = image.resize((self.height, self.width), resample=Image.NEAREST)
                    image = np.asarray(image, dtype=np.uint8)

        if self.transform is not None:
            if mask is not None:
                result = self.transform(image=image, mask=mask)
            else:
                result = self.transform(image=image)
        else:
            if mask is not None:
                result = A.Resize(height=self.height, width=self.width, always_apply=True)(image=image, mask=mask)
            else:
                result = A.Resize(height=self.height, width=self.width, always_apply=True)(image=image)

        #print(result['image'].shape, result['mask'].shape)
        return result

if __name__ == "__main__":
    dd = DRIVE_DATASET('/mnt/d/Datasets/Retinal_Image_Database/DRIVE', 'DRIVE.csv', True)