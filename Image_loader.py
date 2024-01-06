# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, files, hr_shape, device='cuda'):
        hr_height, hr_width = hr_shape
        mean = np.array([0.4488, 0.4371, 0.4040])
        std = np.array([1.0, 1.0, 1.0])
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 2, hr_height // 2), Image.BICUBIC), #BICUBIC
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC), #BICUBIC
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files = files
        
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        img_lr = img_lr.to(device)
        img_hr = img_hr.to(device)
        
            
        return {"lr": img_lr, "hr": img_hr}
    
    def __len__(self):
        return len(self.files)