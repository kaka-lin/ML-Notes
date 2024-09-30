import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # Normalize((12, 12, 12),std = (1, 1, 1)),
    ])


if __name__ == "__main__":
    a = np.arange(1, 26, dtype=np.float32).reshape([5,5])
    a = np.expand_dims(a, 2) # (5, 5, 1)

    data = transform()(a) # (1, 5, 5)
    data = data.unsqueeze(0) # (1, 1, 5, 5)
    print("Input:\n", data.detach().numpy())

    # Standare Conv
    conv1 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=1)
    # Dilated Conv, rate=2, 間隔一格
    conv2 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=2)

    # Initialize weight
    #
    # torch.nn.init.constant_(tensor, val):
    #   Fills the input Tensor with the value val\text{val}val
    nn.init.constant_(conv1.weight, 1)
    nn.init.constant_(conv2.weight, 1)

    result1 = conv1(data)
    result2 = conv2(data)
    print("Standare conv:\n", result1.detach().numpy())
    print("Dilated conv:\n", result2.detach().numpy())
