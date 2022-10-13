import re
import os
import h5py
import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

'''
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

    return data
'''
def read_3d_raw(fileDIR, filename):
    file_url = os.path.join(fileDIR, filename + ".mhd")
    data = io.imread(file_url, plugin='simpleitk')
    data = data.transpose(2, 1, 0)
    return data

def main():
	#===========================================================================
	# Create a HDF5 file.
	f = h5py.File("temp.hdf5", "w")    # mode = {'w', 'r', 'a'}

	Lung_01 = read_3d_raw("C:\\CNN\\3D_UNet_0.1\\Dataset\\training","78950")
	Lung_01_mark = read_3d_raw("C:\\CNN\\3D_UNet_0.1\\Dataset\\training_mark","78950")

	d = f.create_dataset("Lung_01", data=Lung_01)
	t = f.create_dataset("Lung_01_mark", data=Lung_01_mark)

	################
	target_size = 128
	target_block_x = d.shape[0] // target_size
	target_block_y = d.shape[1] // target_size
	tagget_block_z = d.shape[2] // target_size #if necessary

	input_list = []

	for i in range(target_block_x):
		for j in range(target_block_y):
			input_list.append(f.create_dataset("Lung_01_block(%d,%d)" % (i, j), data=Lung_01[target_size*(i):target_size*(i+1),target_size*(j):target_size*(j+1),:]))
			
	# Add two attributes to dataset 'dset'
	d.attrs["name"] = "Lung"
	d.attrs["number"] = "01"
	d.attrs["isRaw"] = True

	t.attrs["name"] = "Lung"
	t.attrs["number"] = "01"
	t.attrs["isRaw"] = False

	# Save and exit the file.
	f.close()

	#===========================================================================
	# Read HDF5 file.
	f = h5py.File("temp.hdf5", "r")    # mode = {'w', 'r', 'a'}

	# Print the keys of groups and datasets under '/'.
	print(f.filename, ":")
	print([key for key in f.keys()], "\n")  

	#===================================================
	# Read dataset 'dset' under '/'.
	d = f["Lung_01"]
	t = f["Lung_01_mark"]

	# Print the data of 'dset'.
	print(d.name, ":")
	print(d.shape)
	print(t.name, ":")
	print(t.shape)

	# Print the attributes of dataset 'dset'.
	for key in d.attrs.keys():
		print(key, ":", d.attrs[key])

	# Save and exit the file
	f.close()

if __name__ == "__main__":
    main()