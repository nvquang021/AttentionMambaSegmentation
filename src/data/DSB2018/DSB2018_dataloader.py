import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8

def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=20, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform
class DataScienceBowl(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = np.load(image_dir)
        self.masks = np.load(mask_dir)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index].squeeze()

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image.type(torch.FloatTensor), mask.unsqueeze(0).type(torch.FloatTensor)
def create_dataloaders(train_img, train_mask, test_img, test_mask, batch_size=BATCH_SIZE):
    train_transform, val_transform = get_transforms()

    train_ds = DataScienceBowl(train_img, train_mask, transform=train_transform)
    test_ds = DataScienceBowl(test_img, test_mask, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, test_loader
