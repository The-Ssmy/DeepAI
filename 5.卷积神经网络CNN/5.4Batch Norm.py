import torch
from torch import nn, optim, transpose

"""
优点：
模型收敛的更快
收敛起来更为稳定
一般流程conv —— batch_norm —— pool —— linear
"""


# 生成所有的channel上面的均值
x = torch.randn(100, 16) + 0.5
layer = nn.BatchNorm1d(16)
print(layer.running_mean, layer.running_var)

for i in range(100):
    out = layer(x)
print(layer.running_mean, layer.running_var)

#
x = torch.rand(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)

out = layer(x)
print(layer.weight, layer.weight.shape)
print(layer.bias, layer.bias.shape)
print(vars(layer))
print(layer.eval())



























