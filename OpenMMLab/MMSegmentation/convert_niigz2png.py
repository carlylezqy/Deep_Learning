import os
import glob
import shutil

import numpy as np
import nibabel as nib
from scipy import ndimage

import mmcv

def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath) # Read file
    scan = scan.get_fdata()   # Get raw data
    return scan

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def task(task_name):
    niigz_path = f"/home/akiyo/cached_dataset/MSD_Raw/{task_name}"
    output_path = f"/home/akiyo/cached_dataset/MSD_FOR_MMSEG/{task_name}"
    
    img_dir = os.path.join(niigz_path, "imagesTr")
    ann_dir = os.path.join(niigz_path, "labelsTr")

    img_suffix = ".nii.gz"

    img_output_dir = os.path.join(output_path, "images")
    ann_output_dir = os.path.join(output_path, "annotations")
    
    shutil.rmtree(img_output_dir, ignore_errors=True); os.makedirs(img_output_dir, exist_ok=True)
    shutil.rmtree(ann_output_dir, ignore_errors=True); os.makedirs(ann_output_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(img_dir, "*" + img_suffix))
    img_paths.sort()

    task_theshold = {
        "Task01_BrainTumour": 1000,
        "Task02_Heart": 100,
        "Task03_Liver": 1000,
        "Task04_Hippocampus": 100,
        "Task05_Prostate": 0,
        "Task06_Lung": 0,
    }

    idx = int(task_name[4:6])

    theshold = task_theshold[task_name]

    for img_path in img_paths:
        ann_path = glob.glob(img_path.replace("imagesTr", "labelsTr"))[0]
        raw_img = read_nifti_file(img_path)
        raw_ann = read_nifti_file(ann_path)

        if   len(raw_img.shape) == 4:
            raw_img = raw_img.transpose(3, 2, 0, 1)
            raw_ann = raw_ann.transpose(2, 0, 1)
            raw_img = raw_img[0]
        elif len(raw_img.shape) == 3:
            raw_img = raw_img.transpose(2, 0, 1)
            raw_ann = raw_ann.transpose(2, 0, 1)

        count = 0
        
        for _img, _ann in zip(raw_img, raw_ann):
            if (_ann.max() + _ann.min()) == 0:
                continue

            ann = np.uint8(_ann)
            #print(_img.shape, np.count_nonzero(ann), ann.max())
            
            if np.count_nonzero(ann) > theshold:
                count += 1

                img = normalization(_img)
                img = (img * 255).astype(np.uint8)

                ann[ann != 0] = idx

                img_name = os.path.basename(img_path).replace(".nii.gz", f"_{count}")
                img_opt_path = os.path.join(img_output_dir, img_name + ".png")
                ann_opt_path = os.path.join(ann_output_dir, img_name + ".png")

                mmcv.imresize(img, (500, 500))
                mmcv.imresize(ann, (500, 500))
                mmcv.imwrite(img, img_opt_path)
                mmcv.imwrite(ann, ann_opt_path)

                #print(ann.max(), end='')
        print(f"\n{os.path.basename(img_path)} FINISH")
def main():
    tasks_name = [
        # "Task01_BrainTumour",
        # "Task02_Heart",
        #"Task03_Liver",
        "Task05_Prostate",
        "Task06_Lung",
    ]
    
    for task_name in tasks_name:
        task(task_name) 

if __name__ == "__main__":
    main()