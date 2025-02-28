import numpy as np
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
import torchvision.datasets as datasets


def load_MNIST(batchsize, classes=None):
    transform = transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(
        "./data",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        "./data",
        download=True,
        train=False,
        transform=transform,
    )

    if classes != None:
        train_dataset.data = train_dataset.data[np.isin(
            train_dataset.targets, classes)]
        train_dataset.targets = train_dataset.targets[np.isin(
            train_dataset.targets, classes)]

        test_dataset.data = test_dataset.data[np.isin(
            test_dataset.targets, classes)]
        test_dataset.targets = test_dataset.targets[np.isin(
            test_dataset.targets, classes)]
    else:
        classes = np.unique(train_dataset.targets)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes

def load_Cifar10(batchsize, classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        "./data",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        "./data",
        download=True,
        train=False,
        transform=transform,
    )
    train_dataset.targets = np.array(train_dataset.targets)
    test_dataset.targets = np.array(test_dataset.targets)

    if classes != None:
        train_dataset.data = train_dataset.data[np.isin(
            train_dataset.targets, classes)]
        train_dataset.targets = train_dataset.targets[np.isin(
            train_dataset.targets, classes)]

        test_dataset.data = test_dataset.data[np.isin(
            test_dataset.targets, classes)]
        test_dataset.targets = test_dataset.targets[np.isin(
            test_dataset.targets, classes)]
    else:
        classes = np.unique(train_dataset.targets)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes
