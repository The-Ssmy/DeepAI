import torch

# 1.view/reshape
# 这两个api是完全一样的
a = torch.randn(4, 1, 28, 28)
a1 = a.view(4, 1, 28*28)
print(a1.shape)
a2 = a.reshape(4, 1, 28*28)
print(a2.shape)
a3 = a.view(1, -1)   # 表示第一个维度是1， 第二个维度是其他维度的集合，也就是[1, 4*1*28*28]
print(a3.shape)

# 2.squeeze/unsqueeze
# 挤压展开维度
b = torch.randn(4, 1, 28, 28)
b1 = b.unsqueeze(0)
b2 = b.unsqueeze(-1)
print(b1.shape)
print(b2.shape)
b3 = b.squeeze(1)   # 只能挤压size为1的位置
print(b3.shape)


# 3.transpose/t/permute
c = torch.rand(3, 4)
c1 = c.t()
print(c1.shape)
c2 = torch.rand(4, 3, 28, 28)
# 维度必须一直跟踪，变换过程中不要影响其物理意义
c3 = c2.transpose(1, 3).contiguous().view(4, 3*28*28).view(4, 28, 28, 3).transpose(1, 3)
c4 = torch.all(torch.eq(c3, c2))
print(c4)
c5 = c2.permute(1, 3, 0, 2)   # 该位置对应其原来的维度
print(c5.shape)

# 4.expand/repeat
# 维度扩展
d0 = torch.rand(4, 32, 14, 14)
d = torch.rand(32)
d1 = d.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
print(d1.shape)
d2 = d1.expand(4, 32, 14, 14)   # 只对原来为1的维度可以进行操作,把原来维度变为给定的参数维度
print(d2.shape)
d3 = d1.expand(-1, 32, -1, 18)
print(d3.shape)

d4 = d1.repeat(4, 1, 14, 14)  # 对应维度重复多少次,不推荐使用
print(d4.shape)











































