import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


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

    # use prefetch_generator and tqdm for iterating through data
    n_batches = len(train_loader)
    with tqdm(train_loader, total=n_batches) as pbar:
        for idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # forward
            pred = model(x)
            loss = loss_fn(pred, y)  # calculate loss

            # backward
            optimizer.zero_grad() # clear gradien
            loss.backward()
            optimizer.step()  # update parameters

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / BATCH_SIZE

            pbar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            pbar.set_postfix(loss=loss.item(), acc=accuracy)
