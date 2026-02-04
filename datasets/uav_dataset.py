"""
PyTorch Dataset for the processed dataset (images resized and masks single-channel)
Assumes directory layout:
  data_dir/
    train/
      images/
      masks/
    val/
      images/
      masks/
"""
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

class UAVDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.names = sorted([p.stem for p in os.scandir(images_dir) if p.name.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.images_dir, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, name + ".png"))
        # Convert to tensors
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        if self.target_transform is None:
            # masks are single-channel with integer labels
            self.target_transform = lambda x: torch.from_numpy(np.array(x, dtype=np.int64))
        img_t = self.transform(img)
        mask_t = self.target_transform(mask)
        return img_t, mask_t