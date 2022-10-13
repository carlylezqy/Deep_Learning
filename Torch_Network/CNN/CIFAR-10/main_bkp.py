import os
import numpy as np
import datatool
import cv2

dataset_name = "cifar-10-batches-py"
image_size = [32, 32]

trainDataTool = datatool.Dataset(dataset_name)
breaken_batch = datatool.break_batch(trainDataTool[0])
labels = trainDataTool.get_labels()
label = labels[breaken_batch['labels'][500]]
data = datatool.array2image(breaken_batch['data'][500], image_size)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])





cv2.imshow(label, data)
cv2.waitKey(0)