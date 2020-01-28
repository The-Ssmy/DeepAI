import torch
from torch import nn

"""
划重点！！！！！！！！！！！
循环神经网络集体实现原理：
    根据参数量建立一个网络层，例如参数量是[10, 3, 100], 就会构造3个单元，每个单元循环10次, 
    这三个单元就可以生成3个长度为10个单词的句子
    喂数据时，单次喂一个维度为[3, 100]的数据，也就是说三个单元各送一个数据


"""


"""
[batch, feature len] @ [hidden len, feature len] + 
[batch, hidden len] @ [hidden len, hidden len]

hidden len:memory表达方式
batch:序列数量
feature len:特征表达方式

forward参数：
    x：实例本身
    h0 = [网络层数， 句子数量， memory表达方式]
    
返回值：
    h：最后一个时间上的所有的memory状态
    out：所有时间上面最后一个memory状态
"""

# 单层
rnn = nn.RNN(100, 10)  # 第一个参数为单词表达方式，第二个参数为memory表达方式
print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)

print(rnn.bias_hh_l0.shape, rnn.bias_ih_l0.shape)


rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x, torch.zeros(1, 3, 20))
print(out.shape, h.shape)



# 双层
rnn = nn.RNN(100, 10, num_layers=2)
print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)

print(rnn.weight_hh_l1.shape, rnn.weight_ih_l1.shape)


# 四层
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x)
print(out.shape, h.shape)


# nn.RNNCell
"""
比起之前那个类这个不会进行多次循环
"""
# 单层
cell1 = nn.RNNCell(100, 20)
h1 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
print(h1.shape)

# 双层
cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
    print(xt.shape)
    print(x.shape)
print(h2.shape)













































