import torch

# 1.加减乘除
a = torch.rand(3, 4)
b = torch.rand(4)
x1 = torch.all(torch.eq(a + b, torch.add(a, b)))  #all方法：所有元素都为true返回true，否则返回false
print(x1)
x2 = torch.all(torch.eq(a - b, torch.sub(a, b)))
print(x2)
x3 = torch.all(torch.eq(a * b, torch.mul(a, b)))
print(x3)
x4 = torch.all(torch.eq(a / b, torch.div(a, b)))
print(x4)

# 2.矩阵相乘
a = torch.rand(2, 3)
b = torch.rand(3, 2)
x1 = torch.all(torch.eq(a @ b, torch.matmul(a, b)))

a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
z = x@w.t()
print(z.shape)


g = torch.rand(4, 3, 512, 784)
x = torch.rand(4, 3, 784,  64)
l = torch.matmul(g, x)
print(l.shape)

# 次方操作
k = torch.full([3, 3], 3)
k1 = k.pow(3)  # 三次方
print(k1)

k2 = k1.sqrt()   # 二次方根,该方法没有参数，只能进行平方根运算
k3 = k1.rsqrt()
print(k2)
print(k3)


# 对数操作
q = torch.exp(torch.full([3, 3], 6))   # 一个3x3数据全为e的6次方的矩阵
print(q)
q1 = torch.log(q)   # 原矩阵以e为底的对数值
print(q1)


# 取整裁剪
w = torch.tensor(3.14)
w1 = w.floor()  # 向下取整
w2 = w.ceil()  # 向上取整
w3 = w.trunc()  # 取整数部分
w4 = w.frac()   # 取小数部分
w5 = w.round()

print(w1, w2, w3, w4, w5)


# 最大最小中间值
t = torch.rand(2, 3)*15
print(t.max(), t.median(), t.min())
t1 = t.clamp(0, 10)  # 0-10之间的数正常显示，大于10的变为10，小于0的变为0
print(t1)






















