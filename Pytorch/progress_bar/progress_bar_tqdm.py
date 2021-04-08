import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Create a simple dataset
x = torch.randn((1000, 3, 224, 224))
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset,  batch_size=8, shuffle=True)

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
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # gradient descent or adam step

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())
