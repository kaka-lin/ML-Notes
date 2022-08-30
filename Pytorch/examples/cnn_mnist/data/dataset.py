from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_data():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = MNIST('./data', download=True, transform=trans)
    test_data = MNIST('./data', train=False, transform=trans)

    return (train_data, test_data)
