import argparse
from pathlib import Path

import torch
from PIL import Image

from models import CatPresenceModel
from trainer import eval_transform


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(
        checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt file found in {checkpoints_dir}")
    return checkpoints[-1]


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
