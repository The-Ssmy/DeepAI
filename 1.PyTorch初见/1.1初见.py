import torch
from torch import autograd
import time
import numpy as np

# 自动求导，对谁求导就把谁看成变量而不是依然把x看成变量
x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**2 * x + b * x + c
print("before:", a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])   # 自动求导函数，返回值为一个列表
print("after:", grads[0], grads[1], grads[2])


# gpu加速
device = torch.device('cuda')
# gpu运算时间为0.29，当然我显卡太差，不然差距会更大
d = torch.rand(50000, 1000).to(device)
e = torch.rand(1000, 2000).to(device)
start_time = time.time()
f = torch.matmul(d, e).to(device)
end_time = time.time()
print("gpu:", end_time - start_time)

# cpu运算时间为1.56，比起gpu慢了不少
h = torch.rand(50000, 1000)
i = torch.rand(1000, 2000)
start_time = time.time()
g = torch.matmul(h, i)
end_time = time.time()
print("cpu:", end_time - start_time)


# 张量数据类型
a = torch.rand(2, 3)
print(a.type())
print(type(a))

device = torch.device('cuda')
a = a.to(device)
print(a.type())
print(type(a))

a = torch.tensor(1.3)  # 0维度,一般用于loss
b = torch.tensor([1.3])   # 1维度，一般用于bias
print(a.shape)
print(b.shape)
print(a.dim())
print(b.dim())


a = torch.tensor([13.3, 22.3])  # .tensor接受的是具体的值
b = torch.FloatTensor(2)  # .FloatTensor接受的是长度
print(b)


data = np.ones(2)
c = torch.from_numpy(data)
print(c.type())

# dim0一般用于loss，为标量
# dim1一般用于bias
# dim2一般用于带有batch的Linear Input输入
# dim3一般用于RNN
# dim4一般用于CNN

x = torch.rand(2, 3, 28, 28)
a.numel()  # 占用内存
a.dim()   # tensor维度
















