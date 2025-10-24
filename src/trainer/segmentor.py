import pytorch_lightning as pl
import torch
from src.trainer.metrics import dice_score, iou_score, precision_score, recall_score, F_score
from src.loss.other_losses import tversky_loss, dice_tversky_loss, focal_tversky, bce_tversky_loss, dice_tversky_loss, FocalLoss
from src.loss.asfm_loss import ASFM_loss

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.criterion = dice_tspd2_loss()  # Replace this with your loss function

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred, decoder_out, layer_out = self.model(image)
        # loss = dice_tversky_loss(y_pred, y_true) +dice_tversky_loss(decoder_out, y_true) +dice_tversky_loss(layer_out, y_true)
        loss = tversky_loss( y_true, y_pred) + tversky_loss( y_true, decoder_out) + tversky_loss(y_true, layer_out)
        # loss = dice_tversky_loss(y_pred, y_true
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        precision = precision_score(y_pred, y_true)
        recall = recall_score(y_pred, y_true)
        f1 = F_score(y_pred, y_true)
        return loss, dice, iou, precision, recall, f1

    def training_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall, f1 = self._step(batch)
        metrics = {
            "loss": loss,
            "train_dice": dice,
            "train_iou": iou,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall, f1 = self._step(batch)
        metrics = {
            "val_loss": loss,
            "val_dice": dice,
            "val_iou": iou,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall, f1 = self._step(batch)
        metrics = {
            "test_loss": loss,
            "test_dice": dice,
            "test_iou": iou,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.5, patience=5, verbose=True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers
