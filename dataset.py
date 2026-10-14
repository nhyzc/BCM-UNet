"""Paired image/mask dataset for crack segmentation."""
import os

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import Config

# Mask files may be stored with any of these extensions.
MASK_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


class CrackDataset(Dataset):
    """Grayscale CT slice paired with a binary crack mask."""

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.exists(image_dir):
            raise FileNotFoundError('Image directory not found: {}'.format(image_dir))
        if not os.path.exists(mask_dir):
            raise FileNotFoundError('Mask directory not found: {}'.format(mask_dir))

        images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith('.png'))
        self.valid_images = []
        for img in images:
            if self._find_mask(os.path.splitext(img)[0]) is not None:
                self.valid_images.append(img)

        if not self.valid_images:
            raise RuntimeError(
                'No image/mask pairs found.\n  images: {}\n  masks : {}'.format(
                    image_dir, mask_dir)
            )
        print('Dataset: {} paired samples loaded from {}'.format(
            len(self.valid_images), image_dir))

    def _find_mask(self, base_name):
        for ext in MASK_EXTS:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self._find_mask(base_name)

        try:
            image = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            mask = (mask > 0.5).float()
            return image, mask

        except Exception as exc:
            print('Error loading {}: {}'.format(img_name, exc))
            return torch.zeros((1, *Config.IMG_SIZE)), torch.zeros((1, *Config.IMG_SIZE))
