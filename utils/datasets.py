import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
    
class CIFAR_SR(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, device="cuda"):
        self.scale_factor = scale_factor
        self.split = split
        self.device = device

        if split == "train":
            train = True
        elif split == "test":
            train = False
        else:
            raise NotImplementedError(f"Unrecognized split: '{split}'")

        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.HR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor))
        ])
        self.LR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

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