import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf


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
train_loader = DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True)

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

# Training
for epoch in range(NUM_EPOCHS):
    # set models to train mode
    model.train()

    n_batches = len(train_loader)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    bar = tf.keras.utils.Progbar(target=n_batches,
                                 stateful_metrics=["loss", "accuracy"])
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)  # calculate loss

        # backward
        optimizer.zero_grad() # clear gradients
        loss.backward()
        optimizer.step()  # update parameters

        # get the index of the max log-probability
        pred = pred.max(1, keepdim=True)[1]
        correct = pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct / BATCH_SIZE

        bar.update(idx,
            values=[("loss", loss.item()), ("accuracy", accuracy)])
    print()
