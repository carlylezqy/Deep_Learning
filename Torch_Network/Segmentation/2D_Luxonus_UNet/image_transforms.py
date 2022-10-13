import numpy as np
from PIL import Image
import cv2
from random import randrange
from torchvision import transforms
import matplotlib.pyplot as plt

def transform(image, mask=None, size=None):
    image = image.convert('L')
    image = cut_to_spin(image)

    if mask is not None:
        image, mask = rotate(image, mask)
        mask = cut_to_spin(mask)

    image = np.array(image)
    # image = histo_equalized(image)
    image = clahe_equalized(image)
    image = adjust_gamma(image, 1.2)

    #image = transforms.ToTensor()(image)
    if mask is not None:
        #mask = np.array(mask)
        #verify(image, mask)
        #mask = transforms.ToTensor()(mask)
        
        return [image, mask]
    else:
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

def rotate(image, mask):
    func = ["false", Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    get_idx = randrange(len(func))
    
    if func[get_idx] != "false":
        #print(mask.shape)
        image = image.transpose(func[get_idx])
        mask = mask.transpose(func[get_idx])
    
    return [image, mask]

def cut_to_spin(image):
    x, y = image.size
    remaining = abs(x - y) // 2
    if x < y:
        image = image.crop((0, remaining, x, y-remaining))
    else:
        image = image.crop((remaining, 0, x-remaining, y))

    x, y = image.size

    if x < y:
        image = image.crop((0, 0, x, y - 1))
    elif x > y:
        image = image.crop((0, 0, x - 1, y))

    return image

def verify(image, mask):
    out = cv2.subtract(image.copy(), mask.copy())
    plt.figure()
    plt.title("verify")
    plt.imshow(out, cmap='gray')
    plt.show()

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