import torch

# cat拼接，要求其他维度必须相同
# stack拼接， 要求所有维度相同扩展出一个新的维度
# split拆分， 按长度拆分，拆成多长的几段
# chunk拆分， 按数量拆分，拆成等分的几份


# 1.拼接
# cat
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
c = torch.cat([a, b], dim=0)   # 这里要保证其他维度具有一致性,也就是非拼接维度其他维度必须相同
print(c.shape)

# stack
a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b], dim=1)   # 添加一个新的维度，相当于把两个相加后以新的维度的方式保存进来，要求所有维度相同
print(c.shape)


# 2.拆分
# split 按长度
a = torch.rand(2, 32, 8)
a1, a2 = a.split(1, dim=0)
print(a1.shape, a2.shape)
a = torch.rand(5, 32, 8)
a1, a2 = a.split([3, 2], dim=0)
print(a1.shape, a2.shape)

# chunk 按数量
a = torch.rand(4, 32, 8)
a1, a2 = a.chunk(2, dim=0)
print(a1.shape, a2.shape)











