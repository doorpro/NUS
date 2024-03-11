'''
coding:utf-8
@software:
@Time:2024/2/26 19:21
@Author:door
'''
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

b = torch.load("F:\\nmrprediction\\NUS\\dataset\\tensordataset\\train_pick.pt")
traindata_loader = DataLoader(b, batch_size=1, shuffle=False)
for i, batch in enumerate(traindata_loader):
    if i == 384:
        plt.subplot(2, 1, 1)
        plt.plot(batch[0].squeeze(0), "b")
        plt.title("label")
        plt.subplot(2, 1, 2)
        plt.plot(batch[1].squeeze(0), "r")
        plt.title("pick")
        plt.show()
        print(batch[0].shape, batch[1].shape)
        break


class deep_picker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 40, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(40, 20, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(20, 10, kernel_size=11, stride=1, padding=5)
        self.conv4 = nn.Conv1d(10, 20, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv1d(20, 10, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(10, 30, kernel_size=11, stride=1, padding=5)
        self.conv7 = nn.Conv1d(30, 18, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.model = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU(), self.conv4,
                                   nn.ReLU(), self.conv5, nn.ReLU(), self.conv6, nn.ReLU(), self.conv7, nn.ReLU(),
                                   self.pool, nn.ReLU())
        self.classifier = nn.Linear(18, 1)
    def forward(self, signal):
        signal = signal.transpose(1, 2)
        x = self.model(signal)
        x = x.transpose(1, 2)
        cls = self.classifier(x)
        return cls


if __name__ == '__main__':
    model = deep_picker()
    a = torch.randn(16, 4096, 1)
    b = model(a)
    print(b.shape)
    print(b)




