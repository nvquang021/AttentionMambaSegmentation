import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from data.PH2.PH2_dataloader import ISICLoader
from src.trainer.metrics import dice_score, iou_score, precision_score, recall_score
from src.loss.other_losses import DiceLoss
from src.models.modules.model import DA_MambaNet
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sklearn.model_selection import train_test_split

model = DA_MambaNet()
data = np.load('./data/Skin_data_192_256/PH2_192_256.npz')
X_train, Y_train = data["image"], data["mask"]
x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=30, random_state=312
)

model.eval()
input = torch.rand(size = (1,3,192,256)).cuda()
flops = FlopCountAnalysis(model.cuda(), input)
print(flop_count_table(flops))

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred, decoder_out, layer_out = self.model(image)
        loss = DiceLoss()(y_pred, y_true)
        print(loss.cpu().numpy(), end = ' ')
        # loss_test.append(loss.item())
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics
    
class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.batch_dice_scores = []
        self.batch_iou_scores = []
        self.batch_precision_scores = []
        self.batch_recall_scores = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred, _, _= self.model(image)

        dice_batch = dice_score(y_pred, y_true)
        iou_batch = iou_score(y_pred, y_true)
        precision_batch = precision_score(y_pred, y_true)
        recall_batch = recall_score(y_pred, y_true)

        self.log_dict({
            "test_dice": dice_batch,
            "test_iou": iou_batch,
            "test_precision": precision_batch,
            "test_recall": recall_batch,
        }, prog_bar=True)

        # Lưu vào mảng
        self.batch_dice_scores.append(dice_batch.item())
        self.batch_iou_scores.append(iou_batch.item())
        self.batch_precision_scores.append(precision_batch.item())
        self.batch_recall_scores.append(recall_batch.item())

    def on_test_end(self):
        print("Dice score per batch:", self.batch_dice_scores)
        print("IoU score per batch:", self.batch_iou_scores)
        print("Precision per batch:", self.batch_precision_scores)
        print("Recall per batch:", self.batch_recall_scores)
        print("STD Dice score:", np.std(self.batch_dice_scores))
        print("STD IoU score:", np.std(self.batch_iou_scores))
        print("STD Precision:", np.std(self.batch_precision_scores))
        print("STD Recall:", np.std(self.batch_recall_scores))

model.eval()
test_dataset = DataLoader(ISICLoader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)
CHECKPOINT_PATH = ""
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)