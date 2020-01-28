import torch
from torch import nn, optim


"""
动量：Momentum
这一步的梯度方向会受上一步梯度方向的影响

学习率衰减：


"""
optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.01, momentum=0.9)

# 方法一
# 固定30步lr变为原来的0.1倍
optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 一般而言step_size都会设置为1k或者10k

for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)

# 方法二
# 多次进行循环操作当多次循环loss没有减少的时候对学习率进行衰减
optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

for epoch in xrange(args.start_epoch, args.epoch):

    train(train_loader, model, criterion, optimizer, epoch)
    result_avg, loss_val = validate(val_loader, model, criterion, epoch)
    scheduler.step(loss_val)  # 每调用一次对梯度进行一次监听，如果没有减少那么会对lr进行操作，如果梯度下降正常则不进行操作
















































