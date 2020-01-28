import  torch
from    torch import nn
from    torch import optim


"""
可以使用很多现成的类

nn.Sequential()容器

.parameters():第一个大维度是第一个参数的weight，第二个维度则是第一个参数的bias，以此类推

net.to(device)

save and load

train/test

实现自己的类
"""

class MyLinear(nn.Module):

    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()

        # requires_grad = True
        self.w = nn.Parameter(torch.randn(outp, inp))  # 可以吧tensor包装到nn.parameters()，梯度信息自动设置为True
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)  # 保留第一个类，其他全打平


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1),
                                 nn.MaxPool2d(2, 2),
                                 Flatten(),
                                 nn.Linear(1*14*14, 10))

    def forward(self, x):
        return self.net(x)

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()

        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(BasicNet(),
                                 nn.ReLU(),
                                 nn.Linear(3, 2))

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device('cuda')
    net = Net()
    net.to(device)

    net.train()

    net.eval()

    # net.load_state_dict(torch.load('ckpt.mdl'))
    #
    #
    # torch.save(net.state_dict(), 'ckpt.mdl')

    for name, t in net.named_parameters():
        print('parameters:', name, t.shape)

    for name, m in net.named_children():
        print('children:', name, m)


    for name, m in net.named_modules():
        print('modules:', name, m)


if __name__ == '__main__':
    main()