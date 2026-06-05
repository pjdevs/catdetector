import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path


LABELS = {
    "vickie": (1, 0),
    "oka": (0, 1),
    "both": (1, 1),
    "none": (0, 0),
}


def session_key(path):
    match = re.match(r"IMG_(\d{8})_(\d{2})(\d{2})\d{2}", path.stem)
    if match:
        day, hour, minute = match.groups()
        return f"{day}-{hour}{minute}"
    return path.stem


def collect_images(source_dir):
    rows = []

    for label_name, (vickie, oka) in LABELS.items():
        label_dir = source_dir / label_name
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label folder: {label_dir}")

        for path in sorted(label_dir.iterdir()):
            if not path.is_file():
                continue

            rows.append(
                {
                    "source_path": path,
                    "filename": path.name,
                    "source_label": label_name,
                    "session": session_key(path),
                    "vickie": vickie,
                    "oka": oka,
                }
            )

    return rows


def split_score(split_counts, targets):
    score = 0
    for split, counts in split_counts.items():
        for key, target in targets[split].items():
            if target == 0:
                continue
            score += ((counts[key] - target) / target) ** 2
    return score


def assign_splits(rows, train_ratio=0.7, val_ratio=0.15):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["session"]].append(row)

    totals = defaultdict(int)
    for row in rows:
        totals["total"] += 1
        totals[row["source_label"]] += 1

    test_ratio = 1 - train_ratio - val_ratio
    targets = {
        "train": {key: value * train_ratio for key, value in totals.items()},
        "val": {key: value * val_ratio for key, value in totals.items()},
        "test": {key: value * test_ratio for key, value in totals.items()},
    }

    split_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    session_groups = []
    for session, items in grouped.items():
        counts = defaultdict(int)
        counts["total"] = len(items)
        for row in items:
            counts[row["source_label"]] += 1
        session_groups.append((session, sorted(items, key=lambda row: row["filename"]), counts))

    session_groups.sort(key=lambda group: (-group[2]["total"], group[0]))

    assigned = {}
    for session, _items, counts in session_groups:
        best_split = None
        best_score = None

        for split in ("train", "val", "test"):
            candidate_counts = {
                name: defaultdict(int, values)
                for name, values in split_counts.items()
            }
            for key, value in counts.items():
                candidate_counts[split][key] += value

            score = split_score(candidate_counts, targets)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assigned[session] = best_split
        for key, value in counts.items():
            split_counts[best_split][key] += value

    split_rows = []
    for row in rows:
        split = assigned[row["session"]]
        copied = dict(row)
        copied["split"] = split
        copied["relative_path"] = str(Path(split) / row["source_label"] / row["filename"]).replace("\\", "/")
        split_rows.append(copied)

    return sorted(split_rows, key=lambda row: (row["split"], row["source_label"], row["filename"]))


def recreate_dataset(rows, output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        for label_name in LABELS:
            (output_dir / split / label_name).mkdir(parents=True, exist_ok=True)

    for row in rows:
        target = output_dir / row["relative_path"]
        shutil.copy2(row["source_path"], target)

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "filename", "relative_path", "vickie", "oka", "source_label", "session"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "vickie": row["vickie"],
                    "oka": row["oka"],
                    "source_label": row["source_label"],
                    "session": row["session"],
                }
            )


def print_summary(rows):
    counts = defaultdict(int)
    sessions = defaultdict(set)

    for row in rows:
        counts[(row["split"], row["source_label"])] += 1
        sessions[row["session"]].add(row["split"])

    print("Counts by split/label:")
    for split in ("train", "val", "test"):
        parts = [f"{label}={counts[(split, label)]}" for label in LABELS]
        print(f"  {split}: " + ", ".join(parts))

    shared_sessions = {session: splits for session, splits in sessions.items() if len(splits) > 1}
    print(f"Shared sessions across splits: {len(shared_sessions)}")
    if shared_sessions:
        for session, splits in sorted(shared_sessions.items()):
            print(f"  {session}: {', '.join(sorted(splits))}")


def main():
    parser = argparse.ArgumentParser(description="Create a fixed train/val/test dataset from label folders.")
    parser.add_argument("--source", default="data", type=Path)
    parser.add_argument("--output", default="dataset", type=Path)
    args = parser.parse_args()

    rows = assign_splits(collect_images(args.source))
    recreate_dataset(rows, args.output)
    print_summary(rows)


if __name__ == "__main__":
    main()
