import os
import random
import glob
import numpy as np
import PIL.Image as pil_image
import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        """
        Args:
            images_dir (str): Directory containing image files.
            patch_size (int): Size of the cropped patches.
            scale (int): Scale factor for super-resolution.
            use_fast_loader (bool): Whether to use TensorFlow's fast image loader.
        """
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        # Load the high-resolution image
        hr = pil_image.open(self.image_files[idx]).convert('RGB')

        # Randomly crop a patch from the high-resolution image
        crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # Generate low-resolution image using bicubic downsampling
        lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        # Convert images to numpy arrays and normalize them to [0, 1]
        hr = np.array(hr).astype(np.float32) / 255.0
        lr = np.array(lr).astype(np.float32) / 255.0

        # Transpose the arrays to (C, H, W) format (channel-first)
        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        return torch.tensor(lr), torch.tensor(hr)

    def __len__(self):
        return len(self.image_files)

# Example usage of the dataset class
if __name__ == "__main__":
    dataset_dir = "C:/Users/IPS/Downloads/RCAN-pytorch/DIV2K/DIV2K_train_HR/x2"  # Replace with your image directory
    patch_size = 48  # Example patch size
    scale_factor = 2  # Example scale factor

    dataset = Dataset(images_dir=dataset_dir, patch_size=patch_size, scale=scale_factor)

    # Get a sample from the dataset
    lr_image, hr_image = dataset[0]
    print(f"Low-Resolution Image Shape: {lr_image.shape}")
    print(f"High-Resolution Image Shape: {hr_image.shape}")
