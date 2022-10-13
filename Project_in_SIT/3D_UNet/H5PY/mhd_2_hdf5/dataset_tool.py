import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from torch.utils.data import Dataset, DataLoader

import os

data_folder = os.path.join(os.getcwd(), "training")

def read_3d_images(fileDIR, filename):
    file_url = os.path.join(fileDIR, filename + ".mhd")
    print(file_url)
    f = open(file_url)
    lines = f.read()
    dimsize = lines[lines.find("DimSize") + 10: lines.find("ElementType") - 1].split()
    f.close()

    DimSize = [int(dimsize[0]), int(dimsize[1]), int(dimsize[2])]

    fname = os.path.join(fileDIR, filename + ".raw")

    data = np.fromfile(fname, '<h')
    data = data.reshape([DimSize[2],DimSize[1],DimSize[0]])
    data = data.transpose(2, 1, 0)

    #data = data[0:300,0:300,:200]
    print(data.shape)

    return data
    ######################################
    '''
    mip = np.max(data, axis=2)
    
    # MIP表示
    plt.figure()
    plt.imshow(mip.transpose(1,0), cmap="gray", vmin=100, vmax=1000) #vminとvmaxはデータごとに調整必要
    plt.colorbar()
    plt.show()
    '''

def read_2d_images(fileDIR):
    return 0

def read_3d_mask_zraw(fileDIR, filename):
    file_url = os.path.join(fileDIR, filename + ".mhd")
    file_url = os.path.join(fileDIR, filename + ".mhd")
    data = io.imread(file_url, plugin='simpleitk')
    data = data.transpose(2, 1, 0)
    print(data.shape)
    return data[0:100,0:100,0:100]

#需要继承torch.utils.data.Dataset
class PBVTrainingDataset(Dataset):
    def __init__(self):
        # 1. Initialize file path or list of file names.
        print("init")

    def __getitem__(self, index): # (2)
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        index = 1
        training = read_3d_images(os.path.join(os.getcwd(), "dataset", "training"), "Signal_1")
        training_mark = read_3d_mask_zraw(os.path.join(os.getcwd(), "dataset", "training_mark"), "78950")

        ###
        training = np.expand_dims(training, axis=0)
        #training = np.expand_dims(training, axis=0)
        ###
        training_mark = np.expand_dims(training_mark, axis=0)
        #training_mark = np.expand_dims(training_mark, axis=0)

        #print(training_mark.shape)

        return {
            'training': torch.from_numpy(training).type(torch.FloatTensor),
            'training_mark': torch.from_numpy(training_mark).type(torch.FloatTensor)
        }
        

    def __len__(self): # (3)
        # You should change 0 to the total size of your dataset.

        return 1
