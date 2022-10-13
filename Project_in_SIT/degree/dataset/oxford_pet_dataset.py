import os
import imageio
import glob
import numpy as np
from PIL import Image

if __name__ == "dataset.oxford_pet_dataset":
    from dataset import general_dataset
elif __name__ == "__main__":
    import general_dataset

class Oxford_Pet(general_dataset.GeneralDataset):
    def __init__(self, image_dir=None, masks_dir=None, image_extension=None, mask_extension=None,
                 albumentation=None, transformer=None, special_rendering=None, train=True):
        image_dir = "/home/akiyo/datasets/The_Oxford-IIIT_Pet_Dataset/images"
        masks_dir = "/home/akiyo/datasets/The_Oxford-IIIT_Pet_Dataset/annotations/trimaps"
        image_extension = ".jpg"; mask_extension = ".png"
        
        def pet_rendering(name, image, mask):
            mask[mask != 1] = 0
            
            if name[0].islower():
                mask = mask * 2

            mask = mask[0] if mask.ndim == 3 else mask
            return image, mask
        
        super(Oxford_Pet, self).__init__(image_dir, masks_dir, image_extension, mask_extension,
                                     albumentation, transformer, pet_rendering, train)
    def decoder(self, mask):
        mask = mask.astype(np.int64)
        color_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_map[mask == 1] = (255, 0, 0) # cat
        color_map[mask == 2] = (0, 255, 0) # dog
        return color_map
    
    @staticmethod
    def get_num_classes():
        return 3