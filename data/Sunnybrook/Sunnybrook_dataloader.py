import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

CROP_SIZE = (160, 160)
BATCH_SIZE = 16

def center_crop(img, crop_size=CROP_SIZE):
    w_in, h_in, d_in = img.shape
    img_crop = np.zeros((*crop_size, d_in))
    w_out, h_out = crop_size

    sub_w = max((w_in - w_out)//2 - 20, 0)
    sub_h = max((h_in - h_out)//2 - 10, 0)

    img_clone = img[sub_w:sub_w + w_out, sub_h:sub_h + h_out]
    img_crop[:img_clone.shape[0], :img_clone.shape[1]] = img_clone
    return img_crop

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
    ])

class SunnyDataset(Dataset):
    def __init__(self, images, masks, transforms=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = center_crop(image)
        mask = center_crop(mask)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.Tensor(mask)
        return image, mask
def create_sunny_dataloaders(train_paths, val_paths, test_paths,
                             batch_size=BATCH_SIZE, transform=True):
    Xtrain = np.load(train_paths[0])
    Ytrain = np.load(train_paths[1])
    Xval = np.load(val_paths[0])
    Yval = np.load(val_paths[1])
    Xtest = np.load(test_paths[0])
    Ytest = np.load(test_paths[1])

    train_transform = get_train_transform() if transform else None

    trainset = SunnyDataset(Xtrain, Ytrain, transforms=train_transform)
    valset = SunnyDataset(Xval, Yval)
    testset = SunnyDataset(Xtest, Ytest)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
