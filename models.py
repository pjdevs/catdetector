import lightning
import torch
import torchmetrics.classification
import torch.nn as nn
from torchvision import models

class CatPresenceModel(lightning.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters()

        # Model

        # MobileNet V2

        # self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # in_features = self.model.classifier[1].in_features
        # self.model.classifier[1] = nn.Linear(in_features, 2)

        # EfficientNe B0

        self.backbone = models.efficientnet_b0(weights=models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2)
        )
        
        # Loss
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        
        # Metrics
        self.acc = torchmetrics.classification.MultilabelAccuracy(num_labels=2, threshold=0.5)

    def forward(self, imgs):
        return self.backbone(imgs)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)

        preds = torch.sigmoid(logits)
        acc = self.acc(preds, labels.int())

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)

        preds = torch.sigmoid(logits)
        acc = self.acc(preds, labels.int())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.backbone.parameters(), lr=self.hparams.lr)
