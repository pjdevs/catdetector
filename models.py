import lightning
import torch
import torchmetrics.classification
import torch.nn as nn
from torchvision import models


class CatPresenceModel(lightning.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        unfreeze_last_block: bool = False,
    ):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.backbone_lr = backbone_lr

        # Model

        # MobileNet V2

        # self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # in_features = self.model.classifier[1].in_features
        # self.model.classifier[1] = nn.Linear(in_features, 2)

        # EfficientNe B0

        self.backbone = models.efficientnet_b0(
            weights=models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_block:
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, 2)
        )

        # Loss
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        # Metrics
        self.acc = torchmetrics.classification.MultilabelAccuracy(
            num_labels=2, threshold=0.5
        )

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
        classifier_params = list(self.backbone.classifier.parameters())
        backbone_params = [
            param
            for name, param in self.backbone.named_parameters()
            if param.requires_grad and not name.startswith("classifier.")
        ]

        params = [{"params": classifier_params, "lr": self.lr}]
        if backbone_params:
            params.append({"params": backbone_params, "lr": self.backbone_lr})

        return torch.optim.Adam(params)
