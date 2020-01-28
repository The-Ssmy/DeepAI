import torch
import torch.nn.functional as F


"""
梯度推导具体过程稍微复杂，梯度值只和输入节点和激活函数以及输出节点有关，所有的权值以mse的输出为纽带
"""
# 单输出感知机
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
print(o.shape)
print(o)

loss = F.mse_loss(torch.ones(1, 1), o)  # 第一个参数为输入，第二个参数为目标
print(loss.shape)

loss.backward()
print(w.grad)




# 多输出感知机
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
print(o.shape)

loss = F.mse_loss(torch.ones(1, 2), o)

loss.backward()
print(w.grad)

























