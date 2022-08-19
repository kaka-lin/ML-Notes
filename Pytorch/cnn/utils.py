import os
import errno

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False


def load_data():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = MNIST('./data', download=True, transform=trans)
    test_data = MNIST('./data', train=False, transform=trans)

    return (train_data, test_data)


def metrics_report_func(x):
    if x is None:
        return 'loss: - acc: '
    if x is not None:
        loss, accuracy = x
        return 'loss: {:.4f} - acc: {:.4f}'.format(loss, accuracy)


def train(model, train_loader, optimizer, loss_fn,
          epoch, epochs, batch_size, device):
    # Sets the model in training mode.
    model.train()

    # use prefetch_generator and tqdm for iterating through data
    n_batches = len(train_loader)
    with tqdm(train_loader, total=n_batches) as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # forward
            pred = model(data)
            loss = loss_fn(pred, target)

            # backward
            optimizer.zero_grad() # clear gradien
            loss.backward()
            optimizer.step()  # update parameters

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / batch_size

            pbar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            pbar.set_postfix(loss=loss.item(), acc=accuracy)


def test(model, test_loader, loss_fn, device):
    # Sets the model in testing mode.
    model.eval()

    test_error = 0
    correct = 0
    n_batches = len(test_loader)

    with tqdm(test_loader, total=n_batches) as pbar:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)

                # forward
                pred = model(data)
                test_error += loss_fn(pred, target).item()

                # get the index of the max log-probability
                pred = pred.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                accuracy = correct / len(test_loader.dataset)
                test_error /= n_batches

                pbar.set_description(f"{batch_idx+1}/{n_batches}")
                pbar.set_postfix(loss=test_error, acc=accuracy)


def save_model(model, path='./', mode='train',
               model_name='model', **kwargs):
    if mode == 'checkpoint':
        path = path + 'models/pre_trains/{}_ckpt.pth'.format(model_name)
    else:
        path = path + 'models/pre_trains/{}.pth'.format(model_name)

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if mode == 'inference':
        torch.save(model.state_dict(), path)
    elif mode == 'checkpoint':
        torch.save({
            'model_state_dict': model.state_dict(),
            **kwargs
        }, path)
    else:
        torch.save(model, path)
