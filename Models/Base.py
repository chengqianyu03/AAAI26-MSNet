import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict

from Utils.metric_utils import dice_metric, iou_metric, hausdorff95_metric
from Utils.utils import to_onehot

class BaseModel(pl.LightningModule):
    def __init__(self, in_channels, out_channels, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = lr
        self.weight_decay = weight_decay

        self.activation_fct = nn.Sigmoid() if self.out_channels == 2 else nn.Softmax(dim=1)
        
        # Define a simple loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def _compute_seg_metrics(self, pred, y):
        onehot_pred = to_onehot(pred.squeeze(1), self.out_channels)
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        dice = torch.mean(dice_metric(onehot_pred, onehot_target))
        iou = torch.mean(iou_metric(onehot_pred, onehot_target))
        hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target))

        metrics = {'dice': 100 * dice, 'iou': 100 * iou, 'hausdorff95': hausdorff95}
        return metrics