import tempfile
import unittest
from os import utime
from pathlib import Path

from catdetector_model.checkpoints import latest_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_latest_checkpoint_returns_most_recent_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            older = temp_dir / "older.ckpt"
            newer = temp_dir / "newer.ckpt"
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")
            utime(older, (1, 1))
            utime(newer, (2, 2))

            self.assertEqual(latest_checkpoint(temp_dir), newer)

    def test_latest_checkpoint_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with self.assertRaisesRegex(FileNotFoundError, "No .ckpt file found"):
                latest_checkpoint(Path(temp_dir_name))


if __name__ == "__main__":
    unittest.main()
