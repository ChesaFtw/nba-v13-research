import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cached_seasons import audit


class CachedSeasonAuditTests(unittest.TestCase):
    def test_complete_seasons(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for season in (2022, 2023):
                path = Path(directory) / f"{season}.json"
                path.write_text(json.dumps({"season": season, "complete": True, "truncated": False}))
                files.append(path)
            self.assertEqual([r["season"] for r in audit(files)], [2022, 2023])

    def test_rejects_missing_explicit_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"season": 2025, "complete": True}))
            with self.assertRaises(ValueError):
                audit([path])

    def test_rejects_duplicate_season(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for name in ("a", "b"):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps({"season": 2024, "complete": True, "truncated": False}))
                files.append(path)
            with self.assertRaises(ValueError):
                audit(files)


if __name__ == "__main__":
    unittest.main()
