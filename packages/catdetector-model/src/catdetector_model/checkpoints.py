from pathlib import Path


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(
        checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt file found in {checkpoints_dir}")
    return checkpoints[-1]
