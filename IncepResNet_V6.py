#!/home/runnanliu/venv/bin python
# @Time     : 2020/05/01
# @Author   : Runnan Liu
# @Python   : 3.5
# @Software : Vscode


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels_branch):
        super(Inception, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels_branch,
                                   kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels_branch,
                                   kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(in_channels, out_channels_branch,
                                   kernel_size=1, stride=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(in_channels, out_channels_branch,
                                   kernel_size=1, stride=1, padding=0)

        self.conv3x3_2 = nn.Conv2d(out_channels_branch, out_channels_branch,
                                   kernel_size=3, stride=1, padding=1)
        self.conv5x5_3 = nn.Conv2d(out_channels_branch, out_channels_branch,
                                   kernel_size=5, stride=1, padding=2)
        self.mpool3x3_4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch_1 = self.conv1x1_1(x)
        branch_2 = self.conv3x3_2(self.conv1x1_2(x))
        branch_3 = self.conv5x5_3(self.conv1x1_3(x))
        branch_4 = self.conv1x1_4(self.mpool3x3_4(x))
        return torch.cat([branch_1, branch_2, branch_3, branch_4], 1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels_branch, out_channels):
        super(ResBlock, self).__init__()
        self.inception = Inception(in_channels, out_channels_branch)
        self.bn1 = nn.BatchNorm2d(out_channels_branch*4)
        self.conv3x3 = nn.Conv2d(out_channels_branch*4, out_channels,
                                 kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.inception(x)
        out = func.leaky_relu(self.bn1(out))
        out = self.bn2(self.conv3x3(out))
        x = self.conv1x1(x)
        return func.leaky_relu(out + x)


class ResGooNet(nn.Module):
    def __init__(self):
        super(ResGooNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
    # self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(16)
        self.resincep1 = ResBlock(16, 8, 64)
        self.resincep2 = ResBlock(64, 32, 256)
        self.resincep3 = ResBlock(256, 128, 1024)
        self.resincep4 = ResBlock(1024, 512, 10)

        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adapavgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 10)

        # self.drop1 = nn.Dropout(0.5)
        # self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        x = func.leaky_relu(self.bn1(self.conv1(x)))
        x = self.mpool1(func.leaky_relu(self.resincep1(x)))
        x = self.mpool2(func.leaky_relu(self.resincep2(x)))
        x = self.mpool3(func.leaky_relu(self.resincep3(x)))
        x = self.adapavgpool(self.resincep4(x))
        x = x.view(-1, 10)

        x = func.log_softmax(x, dim=1)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def label_transform(labels, device):
    labels_tf = torch.zeros(labels.shape[0], 10, dtype=torch.long).to(device)
    row_idx = torch.arange(0, labels.shape[0])
    labels_tf[row_idx, labels] = 1.0
    return labels_tf


def data_test(train_loader, classes, device):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def loss_record(loss_str, record_path):
    f = open(record_path, "a")
    f.write(loss_str)
    f.close()


def train(model, device, train_loader, optimizer,
          loss_func, epoch, interval, loss_str, epls_str):
    model.train()
    num = 0
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # L1 norm regularization
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target) + 1e-5 * regularization_loss
        loss.backward()
        optimizer.step()
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_str = loss_str + str(loss.item()) + "\n"
            num += 1
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / num
    epls_str = epls_str + str(epoch_loss) + "\n"
    print('============================= Loss per epoch : %.6f' % (epoch_loss))
    return loss_str, epls_str


def test(model, device, test_loader, tls_accu_str):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += func.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))

    tls_accu_str = tls_accu_str + str(test_loss) + " " +\
        str(100. * correct / len(test_loader.dataset)) + "\n"
    return tls_accu_str


def train_accu(model, device, train_loader, train_accu_str):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTrain set:  Accuracy: {}/{} ({:.0f}%)\n'.
          format(correct, len(train_loader.dataset),
                 100. * correct / len(train_loader.dataset)))

    train_accu_str = train_accu_str + str(100. * correct /
                                          len(train_loader.dataset)) + "\n"
    return train_accu_str


def class_accuracy(model, device, test_loader, test_batch_size):
    model.eval()
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1, keepdim=True)
            c = predicted.eq(target.view_as(predicted))
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
             classes[i], 100 * class_correct[i] / class_total[i]))


def main():
    # parameter setting
    batch_size = 256
    test_batch_size = 1000
    epochs = 50
    lr = 0.001
    no_cuda = False
    seed = 1
    log_interval = 10
    save_model = True
    root = '/home/runnanliu/BME6840_DIP/Project/data'
    ls_root = "/home/runnanliu/BME6840_DIP/Project/src/IncepResNet_V6"
    # para_pth_rd = "/home/runnanliu/BME6840_DIP/Project/src/" +\
    #               "IncepResNet_V3_OR_CIFAR10_20.pth"
    para_pth_sv = "/home/runnanliu/BME6840_DIP/Project/src/" +\
                  "IncepResNet_V6/IncepResNet_V6_CIFAR10_50.pth"
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=False,
                                            transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batch_size,
                                              shuffle=True, num_workers=6)

    # data_test(train_loader, classes)

    model = ResGooNet().to(device)

    # print('# model parameters:',
    #       sum(param.numel() for param in model.parameters()))
    # model.load_state_dict(torch.load(para_pth_rd))
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # class_accuracy(model, device, test_loader, test_batch_size)

    ls_str = ""
    epls_str = ""
    tls_accu_str = ""
    train_accu_str = ""
    for epoch in range(1, epochs + 1):
        ls_str, epls_str = train(model, device, train_loader, optimizer,
                                 loss_func, epoch, log_interval, ls_str,
                                 epls_str)
        train_accu_str = train_accu(model, device, train_loader,
                                    train_accu_str)
        tls_accu_str = test(model, device, test_loader, tls_accu_str)

    loss_record(ls_str, os.path.join(ls_root, "ls_50.txt"))
    loss_record(epls_str, os.path.join(ls_root, "epls_50.txt"))
    loss_record(tls_accu_str, os.path.join(ls_root, "tls_accu_50.txt"))
    loss_record(train_accu_str, os.path.join(ls_root, "train_accu_50.txt"))

    if save_model:
        torch.save(model.state_dict(), para_pth_sv)


if __name__ == "__main__":
    main()
