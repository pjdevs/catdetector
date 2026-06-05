import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import CatDataset
from models import CatPresenceModel
from trainer import eval_transform


CLASS_NAMES = ("none", "oka", "vickie", "both")
CAT_NAMES = ("vickie", "oka")
DEFAULT_THRESHOLDS = (0.5, 0.5)


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(
        checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt file found in {checkpoints_dir}")
    return checkpoints[-1]


def class_index(label: torch.Tensor) -> int:
    return int(label[0].item()) * 2 + int(label[1].item())


def collect_split_outputs(
    model: CatPresenceModel,
    split: str,
    dataset_dir: Path,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dataset = CatDataset(
        dataset_dir=dataset_dir, split=split, transform=eval_transform()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels_batches = []
    prob_batches = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu()
            labels_batches.append(labels.int())
            prob_batches.append(probs)

    return torch.cat(labels_batches), torch.cat(prob_batches), len(dataset)


def predictions_for_thresholds(
    probs: torch.Tensor, thresholds: tuple[float, float]
) -> torch.Tensor:
    threshold_tensor = torch.tensor(thresholds, dtype=probs.dtype)
    return (probs >= threshold_tensor).int()


def cat_metrics(
    labels: torch.Tensor, preds: torch.Tensor, index: int
) -> tuple[float, float, float, int, int, int, int]:
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        precision,
        recall,
        f1,
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    )


def find_best_thresholds(
    labels: torch.Tensor,
    probs: torch.Tensor,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> tuple[float, float]:
    thresholds = []
    steps = round((max_threshold - min_threshold) / step)
    candidates = [min_threshold + i * step for i in range(steps + 1)]

    for index, cat_name in enumerate(CAT_NAMES):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in candidates:
            pair = DEFAULT_THRESHOLDS
            if index == 0:
                pair = (threshold, pair[1])
            else:
                pair = (pair[0], threshold)

            preds = predictions_for_thresholds(probs, pair)
            _, _, f1, *_ = cat_metrics(labels, preds, index)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)
        print(
            f"best {cat_name} threshold on val: {best_threshold:.2f} f1={best_f1:.3f}"
        )

    return thresholds[0], thresholds[1]


def print_split_report(
    split: str,
    labels: torch.Tensor,
    probs: torch.Tensor,
    sample_count: int,
    thresholds: tuple[float, float],
) -> None:
    preds = predictions_for_thresholds(probs, thresholds)
    loss = torch.nn.functional.binary_cross_entropy(probs, labels.float())

    print(f"\n== {split} ==")
    print(f"images: {sample_count}")
    print(
        "thresholds: "
        + ", ".join(
            f"{name}={threshold:.2f}"
            for name, threshold in zip(CAT_NAMES, thresholds, strict=True)
        )
    )
    print(f"loss: {loss.item():.4f}")
    print(
        "mean_prob: "
        + ", ".join(
            f"{name}={value:.4f}"
            for name, value in zip(CAT_NAMES, probs.mean(dim=0).tolist(), strict=True)
        )
    )

    for index, cat_name in enumerate(CAT_NAMES):
        (
            precision,
            recall,
            f1,
            true_positive,
            false_positive,
            false_negative,
            true_negative,
        ) = cat_metrics(labels, preds, index)

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
    parser.add_argument("--find-thresholds", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    model = CatPresenceModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    thresholds = (args.threshold, args.threshold)
    if args.find_thresholds:
        labels, probs, _ = collect_split_outputs(
            model=model,
            split="val",
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            device=device,
        )
        thresholds = find_best_thresholds(
            labels=labels,
            probs=probs,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
            step=args.threshold_step,
        )
    else:
        print(f"threshold: {args.threshold:.2f}")

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        labels, probs, sample_count = collect_split_outputs(
            model=model,
            split=split,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            device=device,
        )
        print_split_report(
            split=split,
            labels=labels,
            probs=probs,
            sample_count=sample_count,
            thresholds=thresholds,
        )


if __name__ == "__main__":
    main()
