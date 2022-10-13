import os
import imageio
import glob
import numpy as np
from PIL import Image

if __name__ == "dataset.penn_fudan":
    from dataset import general_dataset
elif __name__ == "__main__":
    import general_dataset

class Penn_Fudan(general_dataset.GeneralDataset):
    def __init__(self, image_dir=None, masks_dir=None, image_extension=None, mask_extension=None,
                 albumentation=None, transformer=None, special_rendering=None, train=True):
        image_dir = "/home/akiyo/datasets/PennFudanPed/PNGImages"
        masks_dir = "/home/akiyo/datasets/PennFudanPed/PedMasks"
        image_extension = ".png"; mask_extension = "_mask.png"

        def pf_rendering(name, image, mask):
            mask[mask != 0] = 1
            return image, mask
            
        super(Penn_Fudan, self).__init__(image_dir, masks_dir, image_extension, mask_extension,
                                        albumentation, transformer, pf_rendering, train)
    
    def decoder(self, mask):
        mask = mask * 255
        return mask

    def get_num_classes():
        return 2
        
if __name__ == "__main__":
    dataset = Penn_Fudan()
    image, mask = dataset[0]
    print(image.shape, mask.shape, mask.max(), mask.min())