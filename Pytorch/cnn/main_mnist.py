import os
import errno
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.models import CNN
from utils import *


@click.command()
@click.option('--batch-size', 'batch_size', default=100)
@click.option('--epochs', default=10)
@click.option('--lr', default=0.0001)
@click.option('--save-model', 'saveModel', default=True)
@click.option('--save-model-mode', 'save_model_mode', default='checkpoint')
@click.option('--model-name', 'model_name', default='binary_model')
@click.option('--test-mode', 'testMode', default=False)
def main(batch_size, epochs, lr,
         saveModel, save_model_mode, model_name,
         testMode):

    isCUDA = torch.cuda.is_available()
    # Load data
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss
    model = CNN()
    if isCUDA:
        model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not testMode:
        for epoch in range(0, epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            train_model(model, train_loader, optimizer,
                        loss, batch_size, isCUDA)

        test_model(model, test_loader, loss, batch_size, isCUDA)

        # Save model
        if saveModel:
            checkpoint = {
                'epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            save_model(model, mode=save_model_mode,
                       model_name=model_name, **checkpoint)
    else:
        # Load model
        model = CNN()
        if isCUDA:
            model.cuda()
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        checkpoint = torch.load(
            './models/pre_trains/binary_model_checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        test_model(model, test_loader, loss, batch_size, isCUDA)


if __name__ == "__main__":
    main()
