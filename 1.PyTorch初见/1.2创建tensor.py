import torch
import numpy as np
import pandas as pd



# 从numpy引入
x = np.array([2, 3, 3])
x1 = torch.from_numpy(x)

# 直接输入
x = torch.tensor([2., 32.2])

# 未初始化数据
d = torch.empty(2, 2)
d1 = torch.FloatTensor(2, 2, 3)
d2 = torch.IntTensor(2, 2, 3)


# 转换类型
w = torch.tensor([1.2, 3])
e = torch.tensor([1.555, 2.666])
print(w.type())
print(e.type())
torch.set_default_tensor_type(torch.DoubleTensor)  # 在这行代码之后定义的tensor类型默认为double类型
w1 = torch.tensor([1.2, 3])
e1 = torch.tensor([1.555, 2.666])
print(w1.type())
print(e1.type())


# (0, 1)均匀分布
b = torch.rand(2, 3)
b1 = torch.rand_like(b)
# 自定义均匀分布
v = torch.randint(1, 10, [3, 3])  # 极小值，极大值，shape
v1 = torch.randint_like(v, 1, 10)
print(v1)


# (0, 1)正态分布
c = torch.randn(2, 3)
c1 = torch.randn_like(c)
c2 = torch.normal(mean=torch.full([10], 10), std=torch.arange(1, 0, -0.1))  # 均值方差，使用起来并不直观一般并不适用
print(c2)


# 全都赋值为一个值
t = torch.full([2, 3, 3], 7)
print(t)
t = torch.full([], 7)
print(t)
t = torch.full([1], 7)
print(t)

# 等差数列
y = torch.arange(1, 10)
y1 = torch.arange(1, 10, 2)
print(y1)


# 等比数列
q = torch.linspace(0, 10, 4)  # 从0到10等分的切开
print(q)
q1 = torch.logspace(0, -1, 10)  # 10的n次方



# 几个常用的api
k = torch.ones(2, 3, 3)  # 全为1
k1 = torch.zeros(2, 3, 3)   # 全为0
k2 = torch.eye(3, 4)  # 单位矩阵
print(k2)
k3 = torch.ones_like(k1)


# 可以用同一个随机打散的种子来操作多个tensor，比如学生成绩
o = torch.randperm(2) # 随机打散
print(o)






# 关于tensor维度，先看大环境下有几个小的tensor，再看小的里面有多少比他更小的，以此类推组成一个完整的tensor维度







