import numpy as np
np.random.seed(0)

class regr_layer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.w1 = np.random.normal(0, 1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.normal(0, 1, (hidden_size, output_size))
        self.b2 = np.array([0.0])
        self.lr = learning_rate
    
    def forward(self, x):
        self.x = x
        z1 = np.dot(self.x, self.w1) + self.b1

        self.z1_mask = (z1 > 0)
        self.a1 = z1 * self.z1_mask

        self.y_hat = np.dot(self.a1, self.w2) + self.b2
        return self.y_hat

    def backward(self, y):
        batch_size = y.shape[0]
        dyhat = (self.y_hat - y)/batch_size ## dyhat means dE/dy_hat.
        dw2 = np.dot(self.a1.T, dyhat)
        db2 = np.sum(dyhat, axis=0)
        da1 = np.dot(dyhat, self.w2.T)

        dz1 = da1 * self.z1_mask
        dw1 = np.dot(self.x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
