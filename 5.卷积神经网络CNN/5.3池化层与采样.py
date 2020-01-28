import torch
from torch import nn, optim
import torch.nn.functional as F

# Max pooling
x = torch.rand(1, 16, 14, 14)
layer = nn.MaxPool2d(2, stride=2)   # 第一个参数是kernel_size, 第二个参数是移动步长
print(layer(x).shape)



# Avg pooling
x = torch.rand(1, 16, 14, 14)
layer = nn.AvgPool2d(2, stride=2)
print(layer(x).shape)


# upsample 上采样 放大维度
x = torch.rand(1, 16, 14, 14)
out = F.interpolate(x, scale_factor=2, mode='nearest')  # 第一个参数为被放大数据，第二个为放大倍数，第三个为差值方式
print(out.shape)


# Relu
x = torch.rand(1, 16, 7, 7)
layer = nn.ReLU(True)
out = layer(x)
print(out.shape)









