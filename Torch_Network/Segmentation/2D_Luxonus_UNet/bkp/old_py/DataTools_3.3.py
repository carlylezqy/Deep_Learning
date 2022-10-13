from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
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
        #还是有问题
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
                train_image, train_groud = self.transform(train_image, train_groud, 565)

            return [train_image, train_groud]

        else:
            test_image_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 3])
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
    def __init__(self, dataset_dir, csv_file, batch_size):
        super().__init__()
        self.csv_list = pd.read_csv(os.path.join(dataset_dir, csv_file), header=0)
        self.dataset_dir = dataset_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
    
    def __len__(self):
        return len(max(self.csv_list, self.batch_size)) 
    
    def __getitem__(self, index):
        index = index % self.csv_list
        hdf5_image_path = os.path.join(self.dataset_dir, self.csv_list.iat[index, 0]).replace("/","\\")
        hdf5_image = h5py.File(hdf5_image_path, "r")# mode = {'w', 'r', 'a'}

        output = []
        for key in hdf5_image.keys():
            image = hdf5_image[key][()]
            x, y = image.shape
            image = transforms.ToTensor()(image.reshape(x, y, 1))
            output.append(image)

        hdf5_image.close()

        return output




'''
import image_transforms

transform = image_transforms.transform
qwerty = DRIVE_DATASET("./dataset/DRIVE", "DRIVE.csv", batch_size=50, train=False, transform=transform)
print(qwerty[0])


qwerty = LUXONUS_DATASET("./dataset/luxonus_data", "luxonus_data.csv")
print(len(qwerty[5]))
'''