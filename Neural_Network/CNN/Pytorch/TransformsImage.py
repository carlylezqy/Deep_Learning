import torch as t
import torch.nn as nn
#import torchsnooper
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms.functional as ttf

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('image/weka.jpg')
print(np.asarray(image).shape)
x = ttf.to_tensor(image)
x = ttf.normalize(x, 0.5, 0.5)
print(x.min())

x.unsqueeze_(0)
print(x.shape)
plt.imshow(ttf.to_pil_image(x[0]))
plt.show()