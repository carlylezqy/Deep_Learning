import cv2
import numpy as np

list_opt = []
output = []

for i in range(7):
    this_image = []
    for j in range(8):
        this_image.append(cv2.imread("./output/half1024-0100/block(%d,%d).jpeg" % (i, j), 0))
    list_opt.append(this_image)

for i in range(len(list_opt)):
    image = cv2.hconcat(list_opt[i])
    if i == 0:
        output = np.array(image)
        continue
    output = cv2.vconcat([output, image])

cv2.imwrite("opt.jpg", output)