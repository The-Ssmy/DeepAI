import torch

# 原理

"""
这种数据集是没有标签的
把原数据放入网络先得到一个neck，一般是进行降维，然后再把这个neck送入神经网络，对原数据进行恢复重建
神经网络得到对于训练目标的特征，并且利用这些特征将其恢复，当然也可能出现强行记住的情况

"""


# 变种

"""
PCA主要是一些线性的分析所以会导致其具有很大的局限性，失真较为严重
自编码器得益于激活函数的非线性所以失真较小

Denoising AutoEncoders  添加一些噪声
Dropout AutoEncoders  随机断开一些链接

"""

# 对中间过程h的sample

"""
过程：
    输入 — encoder —> h —decoder —> 输出
在网络中加一个鉴别器，来迫使输出在自己想要的分布

"""

# VAE
"""
可以理解为低配版的GAN
是在生成h(z)之前再加一个神经单元，进行一些非线性的处理
比如KL散度处理，使其在预设的分布内如果不在就往这个方向优化

整体而言VAE效果还不错但是对比GAN简直就是个弟弟

"""









