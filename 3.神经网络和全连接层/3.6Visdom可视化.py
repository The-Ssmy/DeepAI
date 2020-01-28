import torch
from visdom import Visdom

viz = Visdom()
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))
viz.line([[test_loss, correct / len(test_loader.dataset)]], [global_step], win='test', update='append')












