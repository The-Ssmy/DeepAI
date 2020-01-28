import torch
import numpy as np

# 计算平均损失
def computre_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2

    return totalError / float(len(points))


# 计算梯度下降
def step_gradient(b_current, w_current, points, lr):
    """
    这里的w和b可以理解为一个初始化的值，对他进行梯度下降操作来找到更好的w和b
    points是用来训练测试w和b的集合
    lr为学习速率，也就是learning_rate
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)

    return [new_b, new_w]



# 给与初始值，调用梯度下降函数
def gradient_runner(start_b, start_w, lr, points, epochs):
    b = start_b
    w = start_w
    for i in range(epochs):
        b, w = step_gradient(b, w, np.array(points), lr)

    return [b, w]


def main():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    start_b = 1
    start_w = 1
    epochs = 1000
    [b, w] = gradient_runner(start_b, start_w, lr, points, epochs)
    print(b, w)
if __name__ == '__main__':
    main()




















