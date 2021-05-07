import os
import struct

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDatasetFromFile(Dataset):
    """Custom Dataset

    A custom dataset must implement three functions:
    - __init__
    - __len__
    - __getitem__
    """

    def __init__(self, image_file, label_file, transform=None):
        self.image_file = image_file
        self.label_file = label_file
        self.transform = transform

        with open(self.label_file, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            self.labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(self.image_file, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(
                len(self.labels), 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
