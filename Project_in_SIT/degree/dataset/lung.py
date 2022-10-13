import os
import numpy as np
import SimpleITK as sitk
import albumentations as A
from sklearn import preprocessing
from torchvision import transforms
from general_dataset import GeneralDataset

class LUNG16(GeneralDataset):
    def __init__(self, image_dir, masks_dir, image_extension, mask_extension,
                 albumentation=None, transformer=None, special_rendering=None, train=True):
        super(LUNG16, self).__init__(image_dir, masks_dir, image_extension, mask_extension,
                                     albumentation, transformer, special_rendering, train)

    def read_image(self, image_path, mask=False):
        get_extension = lambda image_path: os.path.splitext(os.path.basename(image_path))[-1]
        extension = get_extension(image_path)

        if extension == ".gz" and get_extension(image_path[:-len(extension)]) == ".nii":
            image = sitk.ReadImage(image_path, imageIO="NiftiImageIO")
            image = sitk.GetArrayFromImage(image)
            
        elif extension == ".mhd":
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image)
            
        if not mask:
            image = np.array(image, dtype=np.float32)
        else:
            image = np.array(image, dtype=np.uint8)

        return image


if __name__ == "__main__":
    data_path = "/home/akiyo/datasets/LUNA16/seg-lungs-LUNA16"
    label_path = "/home/akiyo/datasets/LUNA16/subset0"

    height, width = 512, 512

    albumentation = A.Compose([A.Resize(height, width, always_apply=True)])
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataloader = LUNG16(
        image_dir=data_path, masks_dir=label_path, 
        image_extension=".mhd", mask_extension=".mhd", 
        albumentation=albumentation, #transformer=transform,
        #special_rendering=GeneralDataset.special_rendering
    )

    for image, mask in train_dataloader:
        print(image.max(), image.min(), mask.max(), mask.min())
        break
