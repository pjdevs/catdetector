import lightning
import torch
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets import CatDataset
from models import CatPresenceModel


def train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    # Model
    model = CatPresenceModel()

    # Datasets
    train_dataset = CatDataset(split="train", transform=train_transform())
    val_dataset = CatDataset(split="val", transform=eval_transform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=False)

    # Trainer
    logger = TensorBoardLogger("logs", name="cats")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="catdetector-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    trainer = lightning.Trainer(max_epochs=25, logger=logger, callbacks=[checkpoint_callback])

    # Go!
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
