import numpy as np
import torch
from torch import autograd
import matplotlib.pyplot as plt

def f(x):
    return (x-6)**2 + 1

def loss(predictive_value, true_value):
    return torch.mean((predictive_value - true_value) ** 2, dim=0, keepdim=True)

def gradient(W, B, X, Y, loss_function,learning_rate, num_iterations):
    optimizer = torch.optim.Adam([W, B], lr=learning_rate)
    loss_monitor = []
    for i in range(num_iterations):
        optimizer.zero_grad()

        Y_pre = torch.matmul(W.T, X) + B
        
        loss_value = loss_function(Y_pre, Y)
        #grand = autograd.grad(loss_value, [W, B])

        loss_value.backward()
        optimizer.step()

        loss_monitor.append(float(loss_value.detach().numpy()))
        #print('round{}: loss is {}'.format(i, loss_value))
    return [W, B, loss_monitor]

def init_parameter(n_x, n_y):
    return [torch.randn(n_x, n_y, dtype=float, requires_grad=True), torch.randn(n_x, 1, dtype=float, requires_grad=True)]

x = np.arange(-10, 10, 0.1)

y = f(x)

X = torch.tensor(x.reshape(len(x), 1), dtype=float, requires_grad=True)
Y = torch.tensor(y.reshape(len(y), 1), dtype=float, requires_grad=True)
W, B = init_parameter(len(x), 1)

num_iterations = 1000
loss_function = torch.nn.L1Loss()

out = []
lr = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
for i in lr:
    W, B = init_parameter(len(x), 1)
    W, B, loss_monitor = gradient(W, B, X, Y, loss_function, i, num_iterations)
    out.append(np.mean(loss_monitor[-5:]))


plt.ylabel("loss")
plt.plot(range(len(lr)), out)
#print(costs)

#plt.plot(range(num_iterations), costs)
plt.show()
