import torch
from torch.utils.data import Dataset


"""
处理步骤：
    数据集加载：load data
    建立模型：build model
    训练和测试：train and test
    迁移学习：transfer learning

"""

"""
__len__：返回样本数量
__getitem__：返回指定样本

"""

class NumbersDataset(Dataset):

    def __init__(self, training=True):
        if training:
            self.samples = list(range(1, 1001))
        else:
            self.samples = list(range(1001, 1501))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]











