import os
import cv2
import csv
import h5py
import numpy as np
import pandas as pd
import skimage.io as io
import image_transforms
import getMask3d
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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

def train_data():
	target_size = 250
	input_list = []

	transform = image_transforms.transform
	image_url_list = read_image_list(u"C:\ANN\Dataset\Luxonus_Data", "Luxonus_Data.csv")
	name_list = []
	block_list = []
	mask_list = []

	for image_url in image_url_list:
		basename = os.path.basename(image_url).replace(".mhd", "")

		image_3dim = read_3d_raw(image_url)
		image = Dim3_Dim2(image_3dim)
		
		image = (image - np.min(image)) / (np.max(image) - np.min(image))
		image *= 255
		image = image.astype(np.uint8)
		#image = transform(image, size=(image.shape[0], image.shape[1]))
		mask_image = getMask3d.get_mask3d(image)
		cv2.imwrite(basename + ".jpg", mask_image)
		print(image.shape)
		print(mask_image.shape)
		
		h5py_file = h5py.File("c:/ANN/dataset/Luxonus_Data_HDF5/" + basename + ".hdf5", "w")
		target_block_x = image.shape[0] // target_size
		target_block_y = image.shape[1] // target_size
		#tagget_block_z = image.shape[2] // target_size #if necessary

		#h5py_file["image"] = image
		#cv2.imwrite(str(time.time()) + ".jpg", image)
		
		#h5py_file["mask"] = mask_image
		#cv2.imwrite(str(time.time()) + ".jpg", mask_image)

		for i in range(target_block_x):
			for j in range(target_block_y):
				train_image = image[target_size*(i):target_size*(i+1), target_size*(j):target_size*(j+1)]
				train_image_mask = mask_image[target_size*(i):target_size*(i+1), target_size*(j):target_size*(j+1)]
				
				block_name = "block(%d,%d)" % (i, j)
				mask_name = "mask(%d,%d)" % (i, j)

				h5py_file[block_name] = train_image
				h5py_file[mask_name] = train_image_mask

				name_list.append(basename + ".hdf5")
				block_list.append(block_name)
				mask_list.append(mask_name)

				#write into csv file

		h5py_file.attrs["name"] = basename
		h5py_file.attrs["train"] = True
		h5py_file.close()

	dataframe = pd.DataFrame({'file_name':name_list, 'block_list':block_list, 'mask_list': mask_list})
	dataframe.to_csv(u"C:\ANN\Dataset\Luxonus_Data_HDF5\Luxonus_Data_HDF5.csv", index=False, sep=',')

def train_mask():
	print("finish")


def main():
	train_data()
	train_mask()

if __name__ == "__main__":
    main()