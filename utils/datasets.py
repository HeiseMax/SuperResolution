import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# MNIST
class MNIST_SR(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, device="cuda"):
        self.scale_factor = scale_factor
        self.split = split
        self.device = device

        match split:
            case "train":
                train = True
            case "test":
                train = False
            case other:
                raise NotImplementedError(f"Unrecognized split: '{other}'")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        self.HR_dataset = datasets.MNIST(root="./data", train=train, transform=transform, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor))
        ])
        self.LR_dataset = datasets.MNIST(root="./data", train=train, transform=transform_resize, download=download)

    def __len__(self):
        return len(self.HR_dataset)

    def __getitem__(self, index):
        HR, _ = self.HR_dataset[index]
        LR, _ = self.LR_dataset[index]
        # LR = (LR - LR.min()) / (LR.max() - LR.min())
        return HR.to(self.device), LR.to(self.device)
    
    def get_samples(self, n_samples):
        HR_samples = []
        LR_samples = []
        for i in range(n_samples):
            ind = np.random.randint(len(self))
            HR, LR = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0)
    

class MNIST_SR_completion(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, device="cuda"):
        self.scale_factor = scale_factor
        self.split = split
        self.device = device

        match split:
            case "train":
                train = True
            case "test":
                train = False
            case other:
                raise NotImplementedError(f"Unrecognized split: '{other}'")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        self.HR_dataset = datasets.MNIST(root="./data", train=train, transform=transform, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor))
        ])
        self.LR_dataset = datasets.MNIST(root="./data", train=train, transform=transform_resize, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor)),
            transforms.Resize((32, 32))
        ])
        self.LR_up_dataset = datasets.MNIST(root="./data", train=train, transform=transform_resize, download=download)

    def __len__(self):
        return len(self.HR_dataset)

    def __getitem__(self, index):
        HR, _ = self.HR_dataset[index]
        LR, _ = self.LR_dataset[index]
        LR_up, _ = self.LR_up_dataset[index]
        return HR.to(self.device), LR.to(self.device), LR_up.to(self.device)
    
    def get_samples(self, n_samples):
        HR_samples = []
        LR_samples = []
        LR_up_samples = []
        for i in range(n_samples):
            ind = np.random.randint(len(self))
            HR, LR, LR_up = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
            LR_up_samples.append(LR_up)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0), torch.stack(LR_up_samples, dim=0)


# CIFAR    
class CIFAR_SR(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, classes=None, device="cuda"):
        self.scale_factor = scale_factor
        self.split = split
        self.device = device

        match split:
            case "train":
                train = True
            case "test":
                train = False
            case other:
                raise NotImplementedError(f"Unrecognized split: '{other}'")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.HR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor))
        ])
        self.LR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

        if classes != None:
            self.HR_dataset.data = self.HR_dataset.data[np.isin(self.HR_dataset.targets, classes)]
            self.HR_dataset.targets = [1] * len(self.HR_dataset.data)
            self.LR_dataset.data = self.LR_dataset.data[np.isin(self.LR_dataset.targets, classes)]
            self.LR_dataset.targets = [1] * len(self.LR_dataset.data)   

    def __len__(self):
        return len(self.HR_dataset)

    def __getitem__(self, index):
        HR, _ = self.HR_dataset[index]
        LR, _ = self.LR_dataset[index]
        # LR = (LR - LR.min()) / (LR.max() - LR.min())
        return HR.to(self.device), LR.to(self.device)
    
    def get_samples(self, n_samples):
        HR_samples = []
        LR_samples = []
        for i in range(n_samples):
            ind = np.random.randint(len(self))
            HR, LR = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0)

class CIFAR_SR_completion(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, classes=None, device="cuda"):
        self.scale_factor = scale_factor
        self.split = split
        self.device = device

        match split:
            case "train":
                train = True
            case "test":
                train = False
            case other:
                raise NotImplementedError(f"Unrecognized split: '{other}'")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.HR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor))
        ])
        self.LR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor)),
            transforms.Resize((32, 32))
        ])
        self.LR_up_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

        if classes != None:
            self.HR_dataset.data = self.HR_dataset.data[np.isin(self.HR_dataset.targets, classes)]
            self.HR_dataset.targets = [1] * len(self.HR_dataset.data)
            self.LR_dataset.data = self.LR_dataset.data[np.isin(self.LR_dataset.targets, classes)]
            self.LR_dataset.targets = [1] * len(self.LR_dataset.data)
            self.LR_up_dataset.data = self.LR_up_dataset.data[np.isin(self.LR_up_dataset.targets, classes)]
            self.LR_up_dataset.targets = [1] * len(self.LR_up_dataset.data)

    def __len__(self):
        return len(self.HR_dataset)

    def __getitem__(self, index):
        HR, _ = self.HR_dataset[index]
        LR, _ = self.LR_dataset[index]
        LR_up, _ = self.LR_up_dataset[index]
        return HR.to(self.device), LR.to(self.device), LR_up.to(self.device)
    
    def get_samples(self, n_samples):
        HR_samples = []
        LR_samples = []
        LR_up_samples = []
        for i in range(n_samples):
            ind = np.random.randint(len(self))
            HR, LR, LR_up = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
            LR_up_samples.append(LR_up)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0), torch.stack(LR_up_samples, dim=0)
