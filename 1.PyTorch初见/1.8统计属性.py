import torch
# norm表示范数，并不是normalize正则化
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a.norm(1), b.norm(1), c.norm(1))
print(a.norm(2), b.norm(2), c.norm(2))
print(b.norm(1, dim=1))   # 取哪个维度就消掉哪个维度


a = torch.arange(10).view(2, 5)
print(a.sum())
print(a.argmax(dim=0), a.argmin())
print(a.argmax(dim=1, keepdim=True), a.argmin())  # 会保留原来的dim长度
print(a.argmax(dim=0, keepdim=True), a.argmin())


print(a.topk(3, dim=1))   # 返回最大的n个值
print(a.topk(3, dim=1, largest=False))  #返回最小的n个值

print(a.kthvalue(3, dim=1))  # 返回第一维度第3小的值


# 比较运算
a = torch.randn(4, 10)   # 和之前版本不同这里直接返回true或者false
print(a>0, torch.gt(a, 0))  # 同上
print(torch.eq(a, a))  # 用来比较对应位置元素是否相等，返回值为true或者false
print(torch.equal(a, a))  # 用来比较所有元素的值是否相等，返回值同上

























