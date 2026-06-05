import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import CatDataset
from models import CatPresenceModel
from trainer import eval_transform


CLASS_NAMES = ("none", "oka", "vickie", "both")
CAT_NAMES = ("vickie", "oka")


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(
        checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt file found in {checkpoints_dir}")
    return checkpoints[-1]


def class_index(label: torch.Tensor) -> int:
    return int(label[0].item()) * 2 + int(label[1].item())


def evaluate_split(
    model: CatPresenceModel,
    split: str,
    dataset_dir: Path,
    batch_size: int,
    threshold: float,
    device: str,
) -> None:
    dataset = CatDataset(
        dataset_dir=dataset_dir, split=split, transform=eval_transform()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels_batches = []
    preds_batches = []
    prob_batches = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu()
            labels_batches.append(labels.int())
            preds_batches.append((probs >= threshold).int())
            prob_batches.append(probs)

    labels = torch.cat(labels_batches)
    preds = torch.cat(preds_batches)
    probs = torch.cat(prob_batches)
    loss = torch.nn.functional.binary_cross_entropy(probs, labels.float())

    print(f"\n== {split} ==")
    print(f"images: {len(dataset)}")
    print(f"loss: {loss.item():.4f}")
    print(
        "mean_prob: "
        + ", ".join(
            f"{name}={value:.4f}"
            for name, value in zip(CAT_NAMES, probs.mean(dim=0).tolist(), strict=True)
        )
    )

    for index, cat_name in enumerate(CAT_NAMES):
        true_positive = int(((preds[:, index] == 1) & (labels[:, index] == 1)).sum())
        false_positive = int(((preds[:, index] == 1) & (labels[:, index] == 0)).sum())
        false_negative = int(((preds[:, index] == 0) & (labels[:, index] == 1)).sum())
        true_negative = int(((preds[:, index] == 0) & (labels[:, index] == 0)).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        print(
            f"{cat_name}: "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
            f"(tp={true_positive}, fp={false_positive}, "
            f"fn={false_negative}, tn={true_negative})"
        )

    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for label, pred in zip(labels, preds, strict=True):
        confusion[class_index(label)][class_index(pred)] += 1

    print("confusion rows=true cols=pred")
    print("        " + " ".join(f"{name:>7}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, confusion, strict=True):
        print(f"{name:>7} " + " ".join(f"{count:7d}" for count in row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cat detector checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="test"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"threshold: {args.threshold:.2f}")

    model = CatPresenceModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        evaluate_split(
            model=model,
            split=split,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=device,
        )


if __name__ == "__main__":
    main()
