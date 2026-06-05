import torch
from PIL import Image

from models import CatPresenceModel
from trainer import eval_transform


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CatPresenceModel.load_from_checkpoint("checkpoints/catdetector-epoch=23-train_loss=0.5827.ckpt").to(device)
model.eval()

img = Image.open("data/oka/IMG_20260201_161155.jpg").convert("RGB")
img_tensor = torch.unsqueeze(eval_transform()(img), 0).to(device)

with torch.no_grad():
    print(torch.sigmoid(model(img_tensor)))
