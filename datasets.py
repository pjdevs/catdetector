import csv
from pathlib import Path

import torch
from PIL import Image


class CatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir="dataset", split="train", transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        labels_path = self.dataset_dir / "labels.csv"
        with labels_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue

                self.samples.append(
                    (
                        self.dataset_dir / row["relative_path"],
                        [float(row["vickie"]), float(row["oka"])],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float32)

        return img, label
