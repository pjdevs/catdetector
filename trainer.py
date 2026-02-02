import lightning
import torch
import torchvision.transforms as transforms
from models import CatPresenceModel
from datasets import CatDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    # Model
    model = CatPresenceModel()

    # Datasets
    da_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    dataset = CatDataset(
        ["data/vickie", "data/oka", "data/both", "data/none"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        transform=da_transforms
    )
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=False)

    # Trainer
    logger = TensorBoardLogger("logs", name="cats")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="catdetector-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )
    trainer = lightning.Trainer(max_epochs=25, logger=logger, callbacks=[checkpoint_callback])

    # Go!
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
