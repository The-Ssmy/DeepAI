import torch
from torch import nn
import torch.nn.functional as F

x = torch.randn(1, 784)

# 这里参数是ch-in，ch-out
layer1 = nn.Linear(784, 200)
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)

x = layer1(x)
x = F.relu(x, True)
print(x.shape)
x = layer2(x)
print(x.shape)
x = layer3(x)
print(x.shape)



