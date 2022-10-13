import os
import torch
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
def LOAD_CIFAR_BATCH(file):
    with open(file, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj
    
def LOAD_CIFAR_LABELS(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj['label_names']

class CIFAR_DATASET(torch.utils.data.Dataset):
    def __init__(self, transform, ipt_batch_size=5, train=True):
        self.train = train
        self.transform = transform
        self.batch = []

        if train: 
            for i in range(ipt_batch_size):
                url = "C:\\ANN\\Dataset\\cifar-10-batches-py\\data_batch_{}".format(str(i+1))
                self.batch.append(LOAD_CIFAR_BATCH(url))
        else:
            url = "C:\\ANN\\Dataset\\cifar-10-batches-py\\test_batch"
            self.batch.append(LOAD_CIFAR_BATCH(url))

    def __getitem__(self, index):
        new_index = index
        batch_index = 0
        if index < 10000:
            new_index = index
            batch_index = 0
        elif index < 20000:
            new_index = index - 10000
            batch_index = 1
        elif index < 30000:
            new_index = index - 20000
            batch_index = 2
        elif index < 40000:
            new_index = index - 30000
            batch_index = 3
        elif index < 50000:
            new_index = index - 40000
            batch_index = 4
        
        target = self.batch[batch_index][b'labels'][new_index]
        data_raw = self.batch[batch_index][b'data'][new_index]
        data = np.array(data_raw.reshape(3, 32, 32)).transpose(1,2,0)

        if self.transform:
            image = Image.fromarray(data)
            image = self.transform(image)

        return [image, target]

    def get_labels(self):
        url = "C:\\ANN\\Dataset\\cifar-10-batches-py\\batches.meta"
        self.labels = LOAD_CIFAR_LABELS(url)
        return self.labels

    def __len__(self):
        if self.train:
            return 50000
        else:
            return 10000