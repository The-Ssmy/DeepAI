import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

a = torch.tensor([1., 2., 3.])
print(a)
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            
        )


    def forward(self, x):
        pass

