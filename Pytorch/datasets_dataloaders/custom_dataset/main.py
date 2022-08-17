from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from data_tools import load_fashion_mnist
from custom_dataset import CustomDatasetFromFile

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


if __name__ == "__main__":
    # Loading the Fashion-MNIST datasetfrom internet
    load_fashion_mnist()

    # Creating a Custom Dataset for your files
    training_data = CustomDatasetFromFile(
        image_file="data/train-images-idx3-ubyte",
        label_file="data/train-labels-idx1-ubyte",
        transform=ToTensor()
    )

    test_data = CustomDatasetFromFile(
        image_file="data/t10k-images-idx3-ubyte",
        label_file="data/t10k-labels-idx1-ubyte",
        transform=ToTensor()
    )

    # Creating DataLoader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.title(labels_map[label.item()])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
    plt.show()
