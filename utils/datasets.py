import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from PIL import Image
import os
from sklearn.model_selection import train_test_split

class CelebA_SR(Dataset):
    def __init__(self, folder_path="E:\GNN\gnn\datasets\celeba_hq_256", scale_factor=4, split="train", device="cuda"):
        self.device = device
        self.scale_factor = scale_factor
        self.split = split.lower()
        self.image_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg")
        ])

        # Split paths into train/val
        train_paths, val_paths = train_test_split(self.image_paths, test_size=0.2, random_state=42)
        self.data_paths = train_paths if self.split == "train" else val_paths

        # Define transforms
        self.hr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([ 
            transforms.Resize((256 // scale_factor, 256 // scale_factor)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert("RGB")

        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)

        return lr_img.to(self.device), hr_img.to(self.device)

    def get_samples(self, n_samples):
        hr_samples = []
        lr_samples = []
        for i in range(n_samples):
            idx = np.random.randint(len(self)) if self.split == "train" else i
            lr, hr = self[idx]
            hr_samples.append(hr)
            lr_samples.append(lr)
        return torch.stack(hr_samples), torch.stack(lr_samples)


class Pokemon_SR(Dataset):    
    def __init__(self, path="pokemon.npy", scale_factor=4, split="train", device="cuda"):
        self.device = device
        self.scale_factor = scale_factor
        self.split = split

        # Load the dataset
        data = np.load(path).astype(np.float32) / 255.0  # Normalize
        data = torch.tensor(data).permute(0, 3, 1, 2)  # To [N, C, H, W]

        # Split into train/val
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

        self.data = train_data if split == "train" else val_data

        self.hr_transform = transforms.Resize((256, 256))  # or use data.shape[-2:]
        self.lr_transform = transforms.Resize((256 // scale_factor, 256 // scale_factor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hr_img = self.data[idx]  # Tensor: [C, H, W]

        # Convert tensor to PIL for torchvision transforms
        hr_img_pil = transforms.ToPILImage()(hr_img)

        # Apply HR and LR transforms
        hr_img = transforms.ToTensor()(self.hr_transform(hr_img_pil))
        lr_img = transforms.ToTensor()(self.lr_transform(hr_img_pil))

        return lr_img.to(self.device), hr_img.to(self.device)

    def get_samples(self, n_samples):
        hr_samples = []
        lr_samples = []
        for i in range(n_samples):
            idx = np.random.randint(len(self)) if self.split == "train" else i
            lr, hr = self[idx]
            hr_samples.append(hr)
            lr_samples.append(lr)
        return torch.stack(hr_samples), torch.stack(lr_samples)

# MNIST
class MNIST_SR(Dataset):    
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
        return HR.to(self.device), LR.to(self.device)
    
    def get_samples(self, n_samples):
        HR_samples = []
        LR_samples = []
        for i in range(n_samples):
            if self.split == "train":
                ind = np.random.randint(len(self))
            else:
                ind = i
            HR, LR = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0)
    

class MNIST_SR_completion(Dataset):    
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
            if self.split == "train":
                ind = np.random.randint(len(self))
            else:
                ind = i
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
            transforms.Resize((32 // scale_factor, 32 // scale_factor)),
        ])
        self.LR_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

        if classes != None:
            self.HR_dataset.data = self.HR_dataset.data[np.isin(self.HR_dataset.targets, classes)]
            self.HR_dataset.targets = self.HR_dataset.targets[np.isin(self.HR_dataset.targets, classes)]
            self.LR_dataset.data = self.LR_dataset.data[np.isin(self.LR_dataset.targets, classes)]
            self.LR_dataset.targets = self.LR_dataset.targets[np.isin(self.LR_dataset.targets, classes)]
            

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
            if self.split == "train":
                ind = np.random.randint(len(self))
            else:
                ind = i
            HR, LR = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0)

class CIFAR_SR_completion(Dataset):    
    def __init__(self, scale_factor=2, split="train", download=True, classes=None, device="cuda"):
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

        transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32 // scale_factor, 32 // scale_factor)),
            transforms.Resize((32, 32))
        ])
        self.LR_up_dataset = datasets.CIFAR10(root="./data", train=train, transform=transform_resize, download=download)

        if classes != None:
            self.HR_dataset.data = self.HR_dataset.data[np.isin(self.HR_dataset.targets, classes)]
            self.HR_dataset.targets = self.HR_dataset.targets[np.isin(self.HR_dataset.targets, classes)]
            self.LR_dataset.data = self.LR_dataset.data[np.isin(self.LR_dataset.targets, classes)]
            self.LR_dataset.targets =  self.LR_dataset.targets[np.isin(self.LR_dataset.targets, classes)]
            self.LR_up_dataset.data = self.LR_up_dataset.data[np.isin(self.LR_up_dataset.targets, classes)]
            self.LR_up_dataset.targets =  self.LR_up_dataset.targets[np.isin(self.LR_up_dataset.targets, classes)]

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
            if self.split == "train":
                ind = np.random.randint(len(self))
            else:
                ind = i
            HR, LR, LR_up = self[ind]
            HR_samples.append(HR)
            LR_samples.append(LR)
            LR_up_samples.append(LR_up)
        return torch.stack(HR_samples, dim=0), torch.stack(LR_samples, dim=0), torch.stack(LR_up_samples, dim=0)
