import torch
from torch import nn, optim
from    torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from lenet import LeNet
from resnet import ResNet


def main():
    batchsz = 128

    # train
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    # test
    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)


    # x, label = iter(cifar_train).next()
    # print(x.shape, label.shape)

    device = torch.device('cuda')
    model = ResNet().to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)

            logits = model(x)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(epoch, loss.item())

        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(acc)




if __name__ == '__main__':
    main()



















