import torch
from torchvision import datasets, transforms

batch_size = 4
epochs = 4
'''
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3001,))
    ])),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
)

train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
'''
#, generator=torch.Generator().manual_seed(42)
a, b = torch.utils.data.random_split(range(10), [3, 7])
print(a)
print(b.dataset)
#### 使用课件给出的完整py文件表示