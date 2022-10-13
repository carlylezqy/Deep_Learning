import albumentations as A
try:
    from dataset import general_dataset
except ModuleNotFoundError:
    import general_dataset

class Retinal_Database(general_dataset.GeneralDataset):
    def __init__(
        self, image_dir=None, masks_dir=None, image_extension=None, mask_extension=None,
        albumentation=None, transformer=None, special_rendering=None,
        train=True,
    ):  
        data_path = "/home/akiyo/datasets/Retinal_Image_Database/DRIVE/training/images"
        label_path = "/home/akiyo/datasets/Retinal_Image_Database/DRIVE/training/1st_manual"
        image_extension = "_training.tif"; mask_extension = "_manual1.gif"
        self.images_list, self.masks_list = self.load_list(data_path, label_path, image_extension, mask_extension)

        data_path = "/home/akiyo/datasets/Retinal_Image_Database/CHASEDB1/training/images"
        label_path = "/home/akiyo/datasets/Retinal_Image_Database/CHASEDB1/training/mask"
        image_extension=".jpg"; mask_extension="_1stHO.png"
        self.extra_images_list, self.extra_masks_list = self.load_list(data_path, label_path, image_extension, mask_extension)
        self.images_list += self.extra_images_list; self.masks_list += self.extra_masks_list

        data_path = "/home/akiyo/datasets/Retinal_Image_Database/RITE/training/images"
        label_path = "/home/akiyo/datasets/Retinal_Image_Database/RITE/training/vessel"
        image_extension=".tif"; mask_extension=".png"
        self.extra_images_list, self.extra_masks_list = self.load_list(data_path, label_path, image_extension, mask_extension)
        self.images_list += self.extra_images_list; self.masks_list += self.extra_masks_list

        data_path = "/home/akiyo/datasets/Retinal_Image_Database/RITE/test/images"
        label_path = "/home/akiyo/datasets/Retinal_Image_Database/RITE/test/vessel"
        image_extension=".tif"; mask_extension=".png"
        self.extra_images_list, self.extra_masks_list = self.load_list(data_path, label_path, image_extension, mask_extension)
        self.images_list += self.extra_images_list; self.masks_list += self.extra_masks_list

        self.images_list, self.masks_list = self.splitter(image_list=self.images_list, mask_list=self.masks_list, train=train, train_ratio=0.8)

        self.albumentation = albumentation
        self.transfromer = transformer
        def special_rendering(name, image, mask):
            mask[mask==255] = 1
            return image, mask
        self.special_rendering = special_rendering

if __name__ == "__main__":
    height, width = 512, 512
    train_dataset = Retinal_Database(
        albumentation=A.Resize(height, width, always_apply=True), 
        transformer=None,
        train=True
    )

    image, mask = train_dataset[0]
    print(image.shape, mask.shape)