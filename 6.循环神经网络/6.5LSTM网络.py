import torch
import torch.nn as nn


"""
三道门：
    第一道门：用过去的信息(memory)对过去的信息(Ct-1)进行筛选
    第二道门：用过去的信息(memory)对现在的输入(xt)进行筛选
    第三道门：用过去的信息(memory)对前两道门的集合进行筛选
    
"""

"""
nn.LSTM:
    out是全部时间上最后一个memory的状态
    ht和ct代表最后一个时间全部的memory状态
   

nn.LSTMCell
    ht是本次节点的memory状态
    ct是指本次节点的memory状态
    注意：ht和ct之间有着密切联系但并不完全相同，一个是这一层的输出一个是传给下一层的memory，这里姑且都成为memory
    实际上ct是输入到下一层的memory，ht既是这一层的输出也是输入到下一层的memory

"""

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)




# 单层
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])
print(h.shape, c.shape)


# 双层
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])
print(h2.shape, c2.shape)








































