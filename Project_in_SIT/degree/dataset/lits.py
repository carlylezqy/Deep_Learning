import os
import imageio
import glob
import numpy as np
from PIL import Image

if __name__ == "dataset.lits":
    from dataset import general_dataset
elif __name__ == "__main__":
    import general_dataset

class LiTS(general_dataset.GeneralDataset):
    def __init__(self, image_dir=None, masks_dir=None, image_extension=None, mask_extension=None,
                 albumentation=None, transformer=None, special_rendering=None, train=True):
        image_dir = "/home/akiyo/datasets/LITS_Challenge/Training_Batch_1/volume-"
        masks_dir = "/home/akiyo/datasets/LITS_Challenge/Training_Batch_1/segmentation-"
        image_extension = ".nii"; mask_extension = ".nii"
        super(LiTS, self).__init__(image_dir, masks_dir, image_extension, mask_extension,
                                     albumentation, transformer, special_rendering, train)
    
    def load_list(self, image_dir, masks_dir, image_extension, mask_extension):
        images_path = []
        masks_path = []

        for i in range(0, 27):
            images_path.append(glob.glob(image_dir + str(i) + image_extension)[0])
            masks_path.append(glob.glob(masks_dir + str(i) + mask_extension)[0])

        assert len(images_path) > 0, "No valuable images were found."
        assert len(images_path) == len(masks_path), "The number of images and masks must be equal."
        return images_path, masks_path

    def read_image(self, image_path, is_mask=False):
        if os.path.splitext(image_path)[-1] == ".nii":
            image = imageio.v2.imread(image_path)
            image = np.array(image, dtype=np.uint8)
        return image

if __name__ == "__main__":
    lits = LiTS(train=True)
    image, mask = lits[0]
    print(image.shape, mask.max(), mask.min())