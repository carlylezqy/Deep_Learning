import os
import numpy as np
try:
    from dataset.general_dataset import GeneralDataset
except ModuleNotFoundError:
    from general_dataset import GeneralDataset

class VOC2012(GeneralDataset):
    def __init__(self, image_dir=None, masks_dir=None, image_extension=None, mask_extension=None,
                 albumentation=None, transformer=None, special_rendering=None, train=True):
        
        image_dir  = "/mnt/d/Datasets/VOCdevkit/VOC2012/JPEGImages"
        masks_dir = "/mnt/d/Datasets/VOCdevkit/VOC2012/SegmentationClass/pre_encoded"
        txt_path = "/home/akiyo/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation"
        image_extension=".jpg"; mask_extension=".png" 

        self.images_list, self.masks_list = self.load_voc2012(txt_path, image_dir, masks_dir, image_extension, mask_extension)
        
        self.images_list, self.masks_list = self.splitter(image_list=self.images_list, mask_list=self.masks_list, train=train, train_ratio=0.8)

        self.albumentation = albumentation
        self.transfromer = transformer
        self.special_rendering = lambda name, image, mask: (image, mask[0]) if mask.ndim == 3 else (image, mask)
        self.n_classes = 21

    def load_voc2012(self, txt_path, image_dir, masks_dir, image_extension, mask_extension):
        images_list, masks_list = [], []
        
        path = os.path.join(txt_path, "trainval.txt")
        file_list = [file.rstrip() for file in tuple(open(path, "r"))]
        images_list += [os.path.join(image_dir, file + image_extension) for file in file_list]
        masks_list += [os.path.join(masks_dir, file + mask_extension) for file in file_list]

        return images_list, masks_list
    
    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ])

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        mask = mask.astype(int)
        return mask
    
    def decoder(self, mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        mask = np.asarray(mask, dtype=np.uint8)
        label_colours = self.get_pascal_labels()
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for ll in range(0, self.n_classes):
            r[mask == ll] = label_colours[ll, 0]
            g[mask == ll] = label_colours[ll, 1]
            b[mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        
        cmap = np.uint8(rgb * 255)
        
        return cmap

    @staticmethod
    def get_num_classes():
        return 21