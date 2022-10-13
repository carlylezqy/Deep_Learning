import numpy as np
from PIL import Image
import cv2
from random import randrange
from torchvision import transforms

def transform(image, mask=None, size=None):
    image = Image.fromarray(image)
    image = cut_to_spin(image, size[0], size[1])
    image = image.convert('L')
    #image = image.crop((0, 0, size[0], size[1]))
    image = np.array(image)

    ######################
    #image = histo_equalized(image)
    image = clahe_equalized(image)
    image = adjust_gamma(image, 1.2)
    ######################

    image = image.transpose(1, 0)

    if mask is not None:
        mask = np.array(mask)
        image, mask = spin(image, mask)

        return [image, mask]

    return image

def image2gray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray

def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255
    return imgs_normalized

def histo_equalized(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype = np.uint8))
    return imgs_equalized

def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = clahe.apply(np.array(imgs, dtype = np.uint8))
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    return new_imgs


def spin(image, mask):
    func = ["false", 0, 1, -1]
    get_idx = randrange(len(func))
    
    if func[get_idx] != "false":
        #print(mask.shape)
        image = cv2.flip(image, func[get_idx])
        mask = cv2.flip(mask, func[get_idx])
    
    return [image, mask]

def cut_to_spin(image, x, y):
    remaining = abs(x - y) // 2 
    if x < y:
        image = image.crop((0, remaining, x, y - remaining))
    else:
        image = image.crop((remaining, 0, x - remaining, y))

    return image


'''

if __name__ == "__main__":
    image = cv2.imread("./dataset/DRIVE/training/images/21_training.tif")
    mask = cv2.imread("./dataset/DRIVE/training/1st_manual/21_manual1.gif")

    image, mask = spin(image, mask)
    print(mask.shape)

    cv2.imshow("image", image)
    cv2.imshow("image2", mask)

    cv2.waitKey(0)
'''