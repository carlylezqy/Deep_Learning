import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.rand(100, 16, 784)

layer = torch.nn.BatchNorm1d(16)
out = layer(x)

print(layer.running_mean)
print(layer.running_var)

print(out.max(), out.min())

##2D
x = torch.rand(1, 16, 7, 7)
layer = torch.nn.BatchNorm2d(16)
out = layer(x)

print(layer.weight, layer.bias)

#print(vars(layer))
layer.eval()

