import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class RandomCrop(transforms.RandomResizedCrop):
    """Random crop áp dụng đồng thời lên image và mask."""

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(
                imgs[imgCount], i, j, h, w, self.size, self.interpolation
            )
        return imgs
class ISICLoader(Dataset):
    def __init__(self, images, masks, transform=True, typeData="train"):
        self.transform = transform if typeData == "train" else False
        self.typeData = typeData
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)
    def rotate(self, image, mask, degrees=(-15, 15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask

    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        """Kết hợp tất cả các phép augment"""
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
        if self.transform:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.int64)
        mask = torch.from_numpy(mask[np.newaxis])

        return image, mask
if __name__ == "__main__":
    X_train = np.load("/content/.../X_train.npy")
    Y_train = np.load("/content/.../Y_train.npy")
    X_val = np.load("/content/.../X_val.npy")
    Y_val = np.load("/content/.../Y_val.npy")

    # Tạo dataset
    train_dataset = ISICLoader(X_train, Y_train, transform=True, typeData="train")
    val_dataset = ISICLoader(X_val, Y_val, transform=False, typeData="val")

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(len(train_loader.dataset), len(val_loader.dataset))