#v0.14-2

import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
import  xml.dom.minidom
import albumentations as A
from torchvision import transforms
import imageio
import random

from sklearn.utils import shuffle

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_dir, masks_dir, image_extension, mask_extension,
        albumentation=None, transformer=None, special_rendering=None,
        train=True,
    ):
        super(GeneralDataset, self).__init__()

        self.images_list, self.masks_list = self.load_list(image_dir, masks_dir, image_extension, mask_extension)

        self.images_list, self.masks_list = self.splitter(image_list=self.images_list, mask_list=self.masks_list, train=train, train_ratio=0.8)

        self.albumentation = albumentation
        self.transfromer = transformer
        self.special_rendering = special_rendering

    def load_list(self, image_dir, masks_dir, image_extension, mask_extension):
        images_path = glob.glob(os.path.join(image_dir, '*' + image_extension))
        masks_path = []

        for i in range(len(images_path)):
            basename = os.path.basename(images_path[i])[:-len(image_extension)]  + mask_extension
            mask_path = os.path.join(masks_dir, basename)
            if glob.glob(mask_path) is not None:
                masks_path.append(os.path.join(masks_dir, basename))
            else:
                images_path.pop(i)

        assert len(images_path) > 0, "No valuable images were found."
        assert len(images_path) == len(masks_path), "The number of images and masks must be equal."
        return images_path, masks_path

    def read_image(self, image_path, is_mask=False):
        if os.path.splitext(image_path)[-1] == ".ppm":
            image = imageio.v2.imread(image_path)
            image = np.array(image, dtype=np.uint8)
        else:
            image = Image.open(image_path).convert('L' if is_mask else 'RGB')
            image = np.array(image, dtype=np.uint8)
        
        return image

    def __getitem__(self, i):
        image = self.read_image(self.images_list[i])
        mask = self.read_image(self.masks_list[i], is_mask=True)

        if self.albumentation:
            trans = self.albumentation(image=image, mask=mask)
            image, mask = trans['image'], trans['mask']
        
        if self.transfromer:
            image = self.transfromer(image)
            mask = torch.tensor(mask).long() #mask no need Normalize

        if self.special_rendering:
            name = os.path.basename(self.images_list[i]).split('.')[0]
            image, mask = self.special_rendering(name=name, image=image, mask=mask)

        #print(mask.shape)
        return image, mask

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def special_rendering(name, image, mask):
        pass

    @staticmethod
    def encode(mask):
        pass
    
    def decoder(self, mask):
        pass
    
    def append_extra_list(self, image_dir, masks_dir, image_extension, mask_extension):
        extra_images_path, extra_masks_path = self.load_list(image_dir, masks_dir, image_extension, mask_extension)
        self.images_list += extra_images_path
        self.masks_list += extra_masks_path
        return
    
    def splitter(self, image_list, mask_list, train, train_ratio, seed=43):
        total_size = len(image_list)
        train_size = int(total_size * train_ratio)

        image_list, mask_list = shuffle(image_list, mask_list, random_state=seed)

        train_image_list, val_image_list = image_list[:train_size], image_list[train_size:]
        train_mask_list, val_mask_list = mask_list[:train_size], mask_list[train_size:]

        if train:
            return train_image_list, train_mask_list
        else:
            return val_image_list, val_mask_list
    
    @staticmethod
    def get_num_classes():
        raise NotImplementedError

if __name__ == "__main__":
    data_path = "/home/akiyo/datasets/The_Oxford-IIIT_Pet_Dataset/images"
    label_path = "/home/akiyo/datasets/The_Oxford-IIIT_Pet_Dataset/annotations/trimaps"

    height, width = 256, 256

    albumentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.CLAHE(p=1), 
        # A.ElasticTransform(p=1, alpha=1, sigma=23, alpha_affine=40),
        # A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), rotate_limit=0, scale_limit=(-0.2, 0.2), p=1),
        A.Resize(height, width, always_apply=True),
    ])

    transform = transforms.Compose([transforms.ToTensor()])
    # data_path = "/home/akiyo/datasets/Retinal_Image_Database/STARE_Segmentation/Image"
    # label_path = "/home/akiyo/datasets/Retinal_Image_Database/STARE_Segmentation/Mask"
    data_path  = "/mnt/d/Datasets/VOCdevkit/VOC2012/JPEGImages"
    label_path = "/mnt/d/Datasets/VOCdevkit/VOC2012/SegmentationClass/pre_encoded"
    
    voc2012_rendering = lambda name, image, mask: (image, mask[0]) if mask.ndim == 3 else (image, mask)

    train_dataset = GeneralDataset(
        image_dir=data_path, masks_dir=label_path, 
        image_extension=".jpg", mask_extension=".png", 
        albumentation=albumentation, transformer=transform,
        special_rendering=voc2012_rendering,
        train=True
    )
            
    val_dataset = GeneralDataset(
        image_dir=data_path, masks_dir=label_path, 
        image_extension=".jpg", mask_extension=".png", 
        albumentation=A.Resize(height, width, always_apply=True), transformer=transform,
        special_rendering=voc2012_rendering,
        train=False
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
    
    print(len(train_dataset), len(val_dataset))
    
    idx = random.randint(0, len(train_dataset))
    v = train_dataset[idx]
    numpy_image = np.array(v[0], dtype=np.uint8).transpose(1, 2, 0) if transform else np.array(v[0], dtype=np.uint8)
    numpy_mask = np.array(v[1], dtype=np.uint8)

    def voc2012_decode(combine):
        voc_path = "/mnt/d/Datasets/VOCdevkit/VOC2012"
        benchmark = "/mnt/d/Datasets/VOCdevkit/benchmark_RELEASE"
        import voc2012
        voc_dataloader = voc2012.pascalVOCLoader(root=voc_path, sbd_path=benchmark)
        y_cmp = voc_dataloader.decode_segmap(np.asarray(combine, dtype=np.uint8))
        y_cmp = np.uint8(y_cmp * 255)
        return y_cmp

    Image.fromarray(numpy_image).save(f"/home/akiyo/temp/0.jpg")
    Image.fromarray(voc2012_decode(numpy_mask)).save(f"/home/akiyo/temp/0_mask.jpg")