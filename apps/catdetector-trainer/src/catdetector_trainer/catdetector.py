import argparse
from pathlib import Path

import torch
from catdetector_model.checkpoints import latest_checkpoint
from catdetector_model.transforms import eval_transform
from PIL import Image

from catdetector_trainer.models import CatPresenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Vickie/Oka on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CatPresenceModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img_tensor = torch.unsqueeze(eval_transform()(img), 0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor)).squeeze(0).cpu()

    print(f"checkpoint: {checkpoint}")
    print(f"image: {args.image}")
    print(f"vickie: {probs[0].item():.4f} ({probs[0].item() >= args.threshold})")
    print(f"oka: {probs[1].item():.4f} ({probs[1].item() >= args.threshold})")


if __name__ == "__main__":
    main()
