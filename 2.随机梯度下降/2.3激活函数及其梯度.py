import torch


"""
什么是激活函数：
响应值只有在大于某一预值得时候才会发生相应

激活函数分类：
1.sigmoid
2.tanh 多用于循环神经网络RNN
3.relu (可修正线性单元)多用于卷积神经网络CNN


"""
a = torch.linspace(-100, 100, 10)
print(torch.sigmoid(a))

b = torch.linspace(-1, 1, 10)
print(torch.tanh(b))

c = torch.linspace(-1, 1, 10)
print(torch.relu(c))