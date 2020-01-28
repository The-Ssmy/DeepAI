import torch
import numpy as np
import pandas as pd

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)   # 第0张图片的第0个通道
print(a[:2, :1].shape)
print(a[:, :, 0:28:2, 0:28:2].shape)
a1 = a.index_select(0, torch.tensor([0, 1]))
a2 = a.index_select(2, torch.arange(28))

print(a[0, ...].shape)   # 自动扩充其他维度
print(a[0, ..., :1, :26:2].shape)


# select by mask
x = torch.randn(3, 4)
mask = x.ge(0.5)   # 将x中所有大于0.5的值得位置变为1，其他位置变为0
y = torch.masked_select(x, mask)   # 找到x中对应1的位置的所有元素，并将其打平
print(y)


