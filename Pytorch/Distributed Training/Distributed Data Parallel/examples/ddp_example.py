import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    """ Initialize the distributed environment. """
    # Environment variables which need to be
    # set when using c10d's default "env"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = MyModel().to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank])

    # compile the model: optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Start training
    for i in range(1000):
        datas, labels = torch.randn(20, 10).to(rank), torch.randn(20, 5).to(rank)

        # forward pass
        outputs = ddp_model(datas)
        loss = loss_fn(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step() # update parameters

        # if i % 100 == 0:
        #     print("loss: {}".format(loss.item()))

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__=="__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
