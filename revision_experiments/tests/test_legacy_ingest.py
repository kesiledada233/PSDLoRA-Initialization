from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from revision_experiments.scripts.ingest_legacy_recovery import write_manifest


class LegacyIngestTests(unittest.TestCase):
    def test_manifest_is_sorted_and_excludes_itself(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "b").write_text("b\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested/a").write_text("a\n", encoding="utf-8")
            manifest = write_manifest(root)
            lines = manifest.read_text(encoding="utf-8").splitlines()
        self.assertEqual([line.split("  ", 1)[1] for line in lines], ["b", "nested/a"])
        self.assertNotIn("SHA256SUMS", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
