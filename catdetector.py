import torch
import torchvision.transforms as transforms
from models import CatPresenceModel
from PIL import Image

model = CatPresenceModel.load_from_checkpoint("checkpoints/catdetector-epoch=23-train_loss=0.5827.ckpt").to("cuda")
model.eval()

img = Image.open("data/oka/IMG_20260201_161155.jpg").convert("RGB")
resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_tensor = torch.unsqueeze(resize(img), 0).to("cuda")

print(torch.sigmoid(model(img_tensor)))