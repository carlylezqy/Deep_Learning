from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
import image_transforms
import matplotlib.pyplot as plt
import h5py
from torchvision import transforms

class DRIVE_DATASET(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, csv_file, batch_size, train=True, transform=None):
        super().__init__()
        self.csv_list = pd.read_csv(os.path.join(dataset_dir, csv_file), header=0)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.train = train
        self.batch_size = batch_size

    def __len__(self):
        # 还是有问题
        return max(len(self.csv_list), self.batch_size)

    def __getitem__(self, index):
        if self.train:
            train_image_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 0])
            with Image.open(train_image_path) as img:
                train_image = img.convert('RGB')

            train_groud_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 1])
            with Image.open(train_groud_path) as grd:
                train_groud = grd.convert('L')

            if self.transform is not None:
                train_image, train_groud = self.transform(train_image, train_groud, 565)

            return [train_image, train_groud]

        else:
            test_image_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 2])
            with Image.open(test_image_path) as msk:
                test_image = msk.convert('L')

            if self.transform is not None:
                test_image = self.transform(test_image, size=565)

            return test_image


class CHASEDB1_DATASET(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, csv_file, batch_size, train=True, transform=None):
        super().__init__()
        self.csv_list = pd.read_csv(os.path.join(dataset_dir, csv_file), header=0)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.train = train
        self.batch_size = batch_size

    def __len__(self):
        # 还是有问题
        return max(len(self.csv_list), self.batch_size)

    def __getitem__(self, index):
        index = index % len(self.csv_list)
        if self.train:
            train_image_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 0])
            with Image.open(train_image_path) as img:
                train_image = img.convert('RGB')

            train_groud_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 1])
            with Image.open(train_groud_path) as grd:
                train_groud = grd.convert('L')

            if self.transform is not None:
                train_image, train_groud = self.transform(train_image, train_groud, 960)

            return [train_image, train_groud]

class LUXONUS_DATASET(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, csv_file, train, batch_size):
        super().__init__()
        self.csv_list = pd.read_csv(os.path.join(dataset_dir, csv_file))
        self.dataset_dir = dataset_dir
        self.name_list = self.csv_list['file_name'].tolist()
        self.block_list = self.csv_list['block_list'].tolist()
        self.mask_list = self.csv_list['mask_list'].tolist()
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        return max(len(self.name_list), self.batch_size)
    
    def __getitem__(self, index):
        index = index % len(self.name_list)
        hdf5_image_path = os.path.join(self.dataset_dir, self.name_list[index]).replace("/","\\")
        hdf5_image = h5py.File(hdf5_image_path, "r")# mode = {'w', 'r', 'a'}

        image_part = hdf5_image[self.block_list[index]][()]
       
        image_part = np.array(image_part)
        image = image_part.reshape(image_part.shape[0], image_part.shape[1], 1)
        image = transforms.ToTensor()(image)

        if self.train:
            mask_part = hdf5_image[self.mask_list[index]][()]
            mask_part = np.array(mask_part)
            mask = mask_part.reshape(mask_part.shape[0], mask_part.shape[1], 1)
            mask = transforms.ToTensor()(mask)        
        else:
            index = [self.name_list[index].replace(".hdf5",""), self.block_list[index]]

        hdf5_image.close()
        if self.train:
            return [image, mask]
        else:
            return [image, index]

'''
import image_transforms

transform = image_transforms.transform
luxonus = LUXONUS_DATASET(u"C:\ANN\dataset\Luxonus_Data_HDF5", "Luxonus_Data_HDF5.csv", train=False, batch_size=5)

print(luxonus[0][1])


print(len(qwerty.dataset[0][0]),len(qwerty.dataset[0][1]))
for combie in qwerty:
    img, msk = combie
    for i in range(len(img)):
        image_transforms.verify(msk[i][0].numpy(), img[i][0].numpy())
'''
