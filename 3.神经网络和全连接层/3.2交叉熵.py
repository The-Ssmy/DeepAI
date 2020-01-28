import torch
import torch.nn.functional as F

"""
熵：反应不确定性的大小
交叉熵：p(x)logq(x)，也等于熵和散度的和
KL Divergence：散度，看重合情况
"""

a = torch.full([4], 1/4)
print(-(a*torch.log2(a)).sum())

a = torch.tensor([0.1, 0.1, 0.1, 0.7])
print(-(a*torch.log2(a)).sum())

a = torch.tensor([0.001, 0.001, 0.001, 0.997])
print(-(a*torch.log2(a)).sum())



x = torch.rand(1, 784)
w  =torch.rand(10, 784)

logits = x@w.t()
print(F.cross_entropy(logits, torch.tensor([3])))    # 第一个参数直接为矩阵乘就可以，因为这个接口自带了softmax操作


















