import torch
from PIL import Image
from os import listdir
from os.path import isfile, join

class CatDataset(torch.utils.data.Dataset):
    def __init__(self, folder_paths, folder_labels, transform=None):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []

        for img_folder, label in zip(folder_paths, folder_labels):
            for filename in listdir(img_folder):
                file = join(img_folder, filename)

                if isfile(file):
                    self.img_paths.append(file)
                    self.img_labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.img_labels[idx], dtype=torch.float32)

        return img, label
