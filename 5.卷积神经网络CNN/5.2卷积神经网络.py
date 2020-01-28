import torch
from torch import optim, nn


"""
Input_channels:原始图片通道数量
Kernel_channels:核函数数量
Kernel_size:核函数的形状大小
Stride:核函数移动步长
Padding:边缘补丁，在周围补0

"""

# 第一个参数为Input_channels，第二个参数为Kernel_channels
layer = nn.Conv2d(3, 7, kernel_size=3, stride=1, padding=1)
x = torch.rand(10, 3, 28, 28)
print(layer.forward(x).shape)

out = layer(x)
print(out.shape)

print(layer.weight)
print(layer.weight.shape)
print(layer.bias)
print(layer.bias.shape)






