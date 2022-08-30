import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, 3)
        self.conv2 = nn.Conv2d(25, 50, 3)
        self.fc1 = nn.Linear(1250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1250)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw_output
        return x
