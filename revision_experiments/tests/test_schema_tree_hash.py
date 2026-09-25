from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.schema import tree_sha256


class TreeHashTests(unittest.TestCase):
    def test_hash_covers_paths_and_contents_but_not_hidden_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub/b.txt").write_text("two", encoding="utf-8")
            first = tree_sha256(root)
            (root / ".lock").write_text("ignored", encoding="utf-8")
            self.assertEqual(first, tree_sha256(root))
            (root / "sub/b.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(first, tree_sha256(root))


if __name__ == "__main__":
    unittest.main()
