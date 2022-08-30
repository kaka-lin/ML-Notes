import click
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import load_data
from model.model import CNN
from trainer.trainer import train, test
from utils.common import save_model


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
    # Get Device for Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Loading dataset and Creating data loaders
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    if not testMode:
        # Creating the model
        model = CNN().to(device)
        #print(model)

        # Compile the model: optimizer and loss
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            train(model, train_loader,
                  optimizer, loss_fn,
                  epoch, epochs, batch_size, device)

        test(model, test_loader, loss_fn, device)

        # Save model
        if saveModel:
            checkpoint = {
                'epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn
            }
            save_model(model, mode=save_model_mode,
                       model_name=model_name, **checkpoint)
    else:
        # Creating the model
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        checkpoint = torch.load(
            './models/pre_trains/binary_model_ckpt.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch']
        loss_fn = checkpoint['loss']

        test(model, test_loader, loss_fn, device)


if __name__ == "__main__":
    main()
