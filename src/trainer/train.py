import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from src.trainer.segmentor import Segmentor
from src.models.modules.model import DA_MambaNet
from data.PH2.PH2_dataloader import ISICLoader

def load_dataset(dataset_name="PH2"):
    if dataset_name == 'PH2':
        data = np.load('./data/Skin_data_192_256/PH2_192_256.npz')
        X_train, Y_train = data["image"], data["mask"]
        x_train, x_test, y_train, y_test = train_test_split(
            X_train, Y_train, test_size=30, random_state=312
        )
        train_loader = DataLoader(
            ISICLoader(x_train, y_train),
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            prefetch_factor=8
        )
        test_loader = DataLoader(
            ISICLoader(x_test, y_test, typeData="test"),
            batch_size=1,
            num_workers=2,
            prefetch_factor=16
        )
        return train_loader, test_loader
    elif dataset_name == 'DSB':
        data = np.load('./data/DSB_2018.npz')
        X_train, Y_train = data["image"], data["mask"]
        x_train, x_test, y_train, y_test = train_test_split(
            X_train, Y_train, test_size=100, random_state=312
        )
        return x_train, x_test, y_train, y_test
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")

def train_model():
    train_loader, test_loader = load_dataset("PH2")
    model = DA_MambaNet()
    segmentor = Segmentor(model=model)
    os.makedirs('./checkpoint/', exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./checkpoint/',
        filename="ckpt{val_dice:0.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False
    )
    progress_bar = pl.callbacks.TQDMProgressBar()
    trainer = pl.Trainer(
        benchmark=True,
        enable_progress_bar=True,
        logger=True,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=150,
        precision=16,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None
    )
    trainer.fit(segmentor, train_loader, test_loader)

if __name__ == "__main__":
    train_model()
