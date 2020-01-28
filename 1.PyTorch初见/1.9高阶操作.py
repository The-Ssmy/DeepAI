import torch



# where
a = torch.rand(2, 2)
a1 = a.ge(0.5)
print(a1)

b1 = torch.tensor([[1., 2.], [4., 5.]])
b2 = torch.tensor([[0., 0.], [0., 0.]])
b = torch.where(a1, b1, b2)   # 第一个参数为一个选拔矩阵，对应位置为true的位置的值为第二个参数值，false为第二个参数对应位置的值
print(b)


# gather 查表收集
a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
b = torch.tensor([3, 2, 4, 0, 0])
print(torch.gather(a, dim=0, index=b.long()))  # 第一个参数为表本身，第二个参数为查找的维度，第三个参数为查表索引不能为tensor类型


a = torch.randn(4, 10)
idx = a.topk(3, dim=1)  # 其返回值既包含值的大小，又包含值的位置
print(idx)
y = torch.arange(10)
y1 = y + 100
y2 = y1.expand(4, 10)
print(torch.gather(y2, dim=1, index=idx[1].long()))












































