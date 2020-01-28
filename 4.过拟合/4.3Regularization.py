import torch
from torch import nn, optim

"""
正则化：
L1—regularization
L2—regularization
在没有出现过拟合的时候如果使用正则化会导致网络性能急剧下降
因为如果没有出现过拟合说明当前网络的复杂度和数据集的真实复杂度是一致的
因此这种情况下得到的训练结果本来就是刚刚好，正则化相当于强行降低了网络的复杂度

正则化具体原理：
损失函数会随着w参数的越来越多而越来越大，利用w参数的2范数对损失函数进行惩罚，从而网络会在满足训练集的同时选取小一些的w参数
"""

optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.01)  # 这个weight_decay相当于正则化参数












