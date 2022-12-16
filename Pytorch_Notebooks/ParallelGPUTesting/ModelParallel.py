import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:1')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:2')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:1')))
        return self.net2(x.to('cuda:2'))

if __name__ ==  "__main__":
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to('cuda:2')
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(loss)