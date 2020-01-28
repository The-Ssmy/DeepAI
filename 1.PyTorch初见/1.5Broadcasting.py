import torch

# broadcasting自动扩展
# 手动扩展
a = torch.rand(4, 3, 28, 28)
b = torch.tensor([1.])
b1 = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
print(b1.shape)
b2 = b1.expand(4, 3, 28, 28)
print(b2.shape)
b3 = b1.repeat(4, 3, 28, 28)
print(b3.shape)
b4 = a + b3
print(b4.shape)

# 自动扩展
a = torch.rand(4, 3, 28, 28)
b = torch.tensor([1.])
c = a + b
print(c.shape)
d = torch.rand(1, 3)
e = torch.rand(3, 1)
f = d + e
print(f.shape)

































