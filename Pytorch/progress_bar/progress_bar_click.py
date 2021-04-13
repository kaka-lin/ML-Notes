import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import click

from utils import metrics_report_func


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Parameters
NUM_EPOCHS = 10
BATCH_SIZE = 8

# Create a simple dataset
x = torch.randn((1000, 3, 224, 224))
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True)

# Create a simple model
model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3, padding=1, stride=1),
    nn.Flatten(),
    nn.Linear(224*224*10, 10),
).to(device)
print(model)

# Compile the model: optimizer and loss
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
    n_batches = len(loader)
    print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
    with click.progressbar(iterable=loader,
                           label='',
                           show_percent=True, show_pos=True,
                           item_show_func=metrics_report_func,
                           fill_char='=', empty_char='.',
                           width=36) as bar:
        for idx, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # forward
            pred = model(x)
            loss = loss_fn(pred, y)  # calculate loss

            # backward
            loss.backward()
            optimizer.step()  # update parameters

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / BATCH_SIZE

            bar.current_item = [loss, accuracy]
            final_loss = loss
            final_accuracy = accuracy

        bar.current_item = [final_loss, final_accuracy]
        bar.render_progress()
