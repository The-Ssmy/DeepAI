import torch
from pokemon import Pokemon
from utils import Flatten
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':


    batchsz = 6
    # device = torch.device("cuda")
    test_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(test_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512, 5)
                          )
    criteon = nn.CrossEntropyLoss()


    test_db = Pokemon(r'C:\Users\acer\Desktop\DeepAI\宝可梦精灵实战\pokemon', 224, mode='test')
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
    model.load_state_dict(torch.load('best.mdl'))
    for step, (x, y) in enumerate(test_loader):
        if step < 6:
            for i in range(6):
                # x: [b, 3, 224, 224], y: [b]
                model.eval()
                logits = model(x)
                pred = logits.argmax(dim=1)

                print(pred)
                plt.subplot(2,3,i+1)
                plt.imshow(x[i].permute(1, 2, 0))
                plt.suptitle("predict value is {}".format(pred))
            plt.show()


d = {0:"妙蛙种子", 1:"喷火龙", 2:"超梦", 3:"皮卡丘", 4:"杰尼龟"}





















































