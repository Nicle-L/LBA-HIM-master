import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio

class CustomDataset(Dataset):
    def __init__(self, path):
        self.file_list = self.load_file_list(path)
        
    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "Path/to/HSIMix"

        file_list = []
        for line in lines:
            line = line.strip()
            mat_file = os.path.join(root_dir, line)
            file_list.append(mat_file)

        return file_list

    def random_crop(self, img, crop_size):
        """
        Perform a random crop on the input image.
        Args:
            img: (C, H, W) - input image
            crop_size: int - the size of the crop (assume square)
        """
        _, h, w = img.shape
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return img[:, top:top + crop_size, left:left + crop_size]

    def argument(self, img, rotTimes, vFlip, hFlip):
        """
        Apply random rotations and flips on the image
        Args:
            img: (C, H, W) - input image
            rotTimes: number of 90 degree rotations
            vFlip: whether to flip vertically
            hFlip: whether to flip horizontally
        """
        # Random rotations
        for j in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))  # Rotate 90 degrees counterclockwise
        
        # Random vertical flip
        if vFlip:
            img = img[:, :, ::-1]  # Flip along the width axis
        
        # Random horizontal flip
        if hFlip:
            img = img[:, ::-1, :]  # Flip along the height axis
        
        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Load data from .mat file
        mat_file = self.file_list[index]
        input_data = sio.loadmat(mat_file)
        hyper = np.float32(input_data['The Variable name of HSImix'])
        if hyper.shape != (50, 256, 256):
            hyper = np.transpose(hyper, [2, 0, 1])  # Convert to (C, H, W)


        hyper = self.random_crop(hyper, crop_size=256)
        # Random augmentation parameters
        rotTimes = random.randint(0, 3)  # Random rotations (0 to 3)
        vFlip = random.randint(0, 1)  # Random vertical flip (0 or 1)
        hFlip = random.randint(0, 1)  # Random horizontal flip (0 or 1)
        hyper_crop = self.argument(hyper.copy(), rotTimes, vFlip, hFlip)

        # Convert back to numpy and return as torch tensor
        hyper_crop = np.copy(hyper_crop)
        epsilon = 1e-6
        input_min, input_max = np.min(hyper_crop), np.max(hyper_crop)
        input_image = (hyper_crop - input_min) / (input_max - input_min + epsilon)
        input_image_tensor = torch.tensor(input_image, dtype=torch.float32)
        return input_image_tensor


class CustomDataset_Cave(Dataset):
    def __init__(self, path):
        self.file_list = self.load_file_list(path)

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "Path/to/Cave-Data"

        file_list = []
        for line in lines:
            line = line.strip()
            mat_file = os.path.join(root_dir, line)
            file_list.append(mat_file)

        return file_list

    def random_crop(self, img, crop_size):
        """
        Perform a random crop on the input image.
        Args:
            img: (C, H, W) - input image
            crop_size: int - the size of the crop (assume square)
        """
        _, h, w = img.shape
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return img[:, top:top + crop_size, left:left + crop_size]

    def argument(self, img, rotTimes, vFlip, hFlip):
        """
        Apply random rotations and flips on the image
        Args:
            img: (C, H, W) - input image
            rotTimes: number of 90 degree rotations
            vFlip: whether to flip vertically
            hFlip: whether to flip horizontally
        """
        # Random rotations
        for j in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))  # Rotate 90 degrees counterclockwise

        # Random vertical flip
        if vFlip:
            img = img[:, :, ::-1]  # Flip along the width axis

        # Random horizontal flip
        if hFlip:
            img = img[:, ::-1, :]  # Flip along the height axis

        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Load data from .mat file
        mat_file = self.file_list[index]
        input_data = sio.loadmat(mat_file)

        hyper = np.float32(input_data['The Variable name of Cave'])
        # 检查维度是否需要转换
        if hyper.shape != (50, 512, 512):  
            hyper = np.transpose(hyper, [2, 0, 1])  # Convert to (C, H, W)

        hyper = self.random_crop(hyper, crop_size=256)
        # Random augmentation parameters
        rotTimes = random.randint(0, 3)  # Random rotations (0 to 3)
        vFlip = random.randint(0, 1)  # Random vertical flip (0 or 1)
        hFlip = random.randint(0, 1)  # Random horizontal flip (0 or 1)
        hyper_crop = self.argument(hyper.copy(), rotTimes, vFlip, hFlip)

        # Convert back to numpy and return as torch tensor
        hyper_crop = np.copy(hyper_crop)
        epsilon = 1e-6
        input_min, input_max = np.min(hyper_crop), np.max(hyper_crop)
        input_image = (hyper_crop - input_min) / (input_max - input_min + epsilon)

        input_image_tensor = torch.tensor(input_image, dtype=torch.float32)
        return input_image_tensor
