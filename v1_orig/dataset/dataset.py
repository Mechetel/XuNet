"""This module provides the data samples for training/validation."""

import os
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import imageio as io


class DatasetLoad(Dataset):
    """Dataset for loading cover/stego pairs."""
    def __init__(self, cover_path, stego_path, mode, transform=None):
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform

        if mode == "train":
            self.indices = list(range(1, 8001))
        elif mode == "val":
            self.indices = list(range(8001, 10001))
        else:
            raise ValueError("mode must be 'train' or 'val'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img_name = f"{index}.pgm"

        cover_img = io.imread(os.path.join(self.cover, img_name))
        stego_img = io.imread(os.path.join(self.stego, img_name))

        # Normalize to [0, 1]
        cover_img = cover_img.astype('float32') / 255.0
        stego_img = stego_img.astype('float32') / 255.0

        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        return {
            "cover": cover_img,
            "stego": stego_img,
        }
