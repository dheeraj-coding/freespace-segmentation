import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        folders = os.listdir(path)
        self.img_names = [os.path.join(path, folder, name) for folder in folders
                          for name in os.listdir(os.path.join(path, folder))]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        x = Image.open(self.img_names[index])
        xo = x.copy()

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            xo = self.target_transform(xo)
        xo = np.array(xo)
        return x, xo


def get_dataloader(path, scale_size=256, crop_size=224, batch_size=4, shuffle=True, num_workers=4):
    transformer = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])
    target_transformer = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(crop_size),
    ])

    dataset = ImageDataset(path, transformer, target_transformer)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
