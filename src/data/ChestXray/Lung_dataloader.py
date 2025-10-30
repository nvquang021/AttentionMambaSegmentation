import os
import numpy as np
import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.backends.cudnn as cudnn


def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])


def load_images(data_path, split='train'):
    data = np.load(data_path)
    return data[f"{split}_img"]


def load_masks(data_path, split='train'):
    data = np.load(data_path)
    masks = data[f"{split}_msk"]
    return masks.squeeze(-1)


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform is not None:
            transformed = self.transform(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

        img = transforms.ToTensor()(img)
        msk = np.expand_dims(msk, axis=-1)
        msk = transforms.ToTensor()(msk)

        return img, msk

    def __len__(self):
        return len(self.images)


def create_dataloaders(train_path, test_path, batch_size=8, num_workers=2):
    x_train = np.load(os.path.join(train_path, "X_train.npy"))
    y_train = np.load(os.path.join(train_path, "y_train.npy"))
    x_test = np.load(os.path.join(test_path, "X_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))

    y_train = y_train[:, :, :, 0]
    y_test = y_test[:, :, :, 0]
    x_val, y_val = x_test, y_test

    # Dataset
    train_dataset = SegmentationDataset(x_train, y_train, transform=get_train_transform())
    val_dataset = SegmentationDataset(x_val, y_val, transform=get_val_transform())
    test_dataset = SegmentationDataset(x_test, y_test, transform=get_val_transform())

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
