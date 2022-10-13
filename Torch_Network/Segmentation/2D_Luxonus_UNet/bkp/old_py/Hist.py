import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import h5py
import torch
import cv2
import numpy as np
import pandas as pd
import skimage.io as io
import image_transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import cv2

def read_image_list(dataset_dir, csv_file):
	image_url_list = []
	csv_list = pd.read_csv(os.path.join(dataset_dir, csv_file), header=0)
	for i in range(len(csv_list)):
		image_path = os.path.join(dataset_dir, csv_list.iat[i, 0]).replace("/", "\\")
		image_url_list.append(image_path)
	
	return image_url_list

def read_3d_raw(file_url):
    data = io.imread(file_url, plugin='simpleitk')
    data = data.transpose(2, 1, 0)
    return data

def Dim3_Dim2(image):
	image = np.max(image, axis=2).transpose(1, 0)
	image = image.astype(np.uint16)
	return image

image_url_list = read_image_list(u"C:\CNN\Dataset\Luxonus_Data", "Luxonus_Data.csv")

for image_url in image_url_list:
    image_full_3dim = read_3d_raw(image_url)
    image_full = Dim3_Dim2(image_full_3dim)

    image_full = (image_full - np.min(image_full)) / (np.max(image_full) - np.min(image_full))
    image_full *= 255
    image_full = image_full.astype(np.uint8)


    #cv2.imshow("Original", image_full)
    #cv2.waitKey(0)

    hist = cv2.calcHist([image_full],[0],None,[256],[0,255])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0,255])
    plt.show()