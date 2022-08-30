import torch
from tqdm import tqdm


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
