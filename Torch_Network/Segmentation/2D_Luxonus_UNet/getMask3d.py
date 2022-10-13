import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.io as io
import os
import cv2
import time

def get_clean_image(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    output = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for i in range(1, num_labels):
        if stats[i, 4] > 10:
            mask = labels == i
            output[:, :][mask] = 255

    return output


def get_mask3d(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image[image < 0.2] = 0
    image[image >= 0.2] = 1
    image *= 255
    image = image.astype(np.uint8)
    image = get_clean_image(image)

    return image