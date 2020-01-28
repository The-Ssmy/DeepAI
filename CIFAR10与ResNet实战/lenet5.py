import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10)
        )

        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):

        batchsz = x.size[0]
        x = self.conv_unit(x)
        x = x.view(batchsz, 16*5*5)
        logits = self.fc_unit(x)

        # loss = self.criteon(logits, y)
        return logits
















