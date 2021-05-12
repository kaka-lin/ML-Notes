import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os, sys, random, time
import argparse

from focal_loss_torch import *


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

start_time = time.time()
max_error = 0
for i in range(1000):
    preds = torch.rand(12800, 2) * random.randint(1, 10)
    preds = Variable(preds.to(device))
    # Whether >= 0.1 or not
    labels = torch.rand(12800).ge(0.1).long()
    labels = Variable(labels.to(device))

    focal_loss = FocalLoss(gamma=0)(preds, labels)
    cross_entropy_loss = nn.CrossEntropyLoss()(preds, labels)

    a = focal_loss.item()
    b = cross_entropy_loss.item()
    if abs(a-b) > max_error:
        max_error = abs(a-b)
print('time:', time.time() - start_time, ', max error:', max_error)


start_time = time.time()
max_error = 0
for i in range(100):
    preds = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
    preds = Variable(preds.to(device))
    labels = torch.rand(128,8,4) * 1000    # 1000 is classes_num
    labels = labels.long()
    labels = Variable(labels.to(device))

    focal_loss = FocalLoss(gamma=0)(preds, labels)
    nll_loss = nn.NLLLoss2d()(F.log_softmax(preds), labels)

    a = focal_loss.item()
    b = nll_loss.item()
    if abs(a-b) > max_error:
        max_error = abs(a-b)
print('time:', time.time() - start_time, ', max_error:', max_error)
