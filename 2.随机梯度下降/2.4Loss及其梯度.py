import torch
from torch.nn import functional as F


"""
1.Mean Squared Error MSE
均方差  多用于回归问题，另外提一句对于分类问题一般使用交叉熵

2.Softmax
扩大差距倍数，一般用于概率多分类问题

"""
# 1
x = torch.ones(1)
w = torch.full([1], 2, requires_grad=True)
mse = F.mse_loss(torch.ones(1), x*w)  # 第一个参数为输入，第二个参数为目标
print(torch.autograd.grad(mse, [w]))


x = torch.ones(1)
w = torch.full([1], 2)
w.requires_grad_()
mse = F.mse_loss(w*x, torch.ones(1))
mse.backward()
print(w.grad)



# 2
a = torch.tensor([1., 2., 3.], requires_grad=True)
p = F.softmax(a, dim=0)
print(p)
print(torch.autograd.grad(p[1], [a], retain_graph=True))
print(torch.autograd.grad(p[2], [a], retain_graph=True))








