import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.json_io import write_json


class AtomicJsonTests(unittest.TestCase):
    def test_syncs_before_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            with patch("common.json_io.os.fsync") as sync:
                write_json(path, {"completed": ["page1"]})
            sync.assert_called_once()
            self.assertEqual(json.loads(path.read_text()), {"completed": ["page1"]})

    def test_crash_during_write_or_replace_preserves_previous_checkpoint(self):
        for operation in ("common.json_io.json.dump", "common.json_io.os.fsync", "common.json_io.Path.replace"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.json"
                write_json(path, {"completed": ["page1"]})
                with patch(operation, side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                    write_json(path, {"completed": ["page1", "page2"]})
                self.assertEqual(json.loads(path.read_text()), {"completed": ["page1"]})
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])
