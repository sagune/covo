import sys
import tempfile
import unittest
from pathlib import Path

from covo.config import build_command, command_to_string, run_pipeline


class ConfigTest(unittest.TestCase):
    def test_build_command(self):
        command = build_command(
            {
                "script": "scripts/x.py",
                "args": {
                    "input": "a.jsonl",
                    "dry_run": True,
                    "skip": False,
                    "topk": [1, 2],
                },
            }
        )
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--input", command)
        self.assertIn("--dry-run", command)
        self.assertNotIn("--skip", command)
        self.assertIn("1", command)
        self.assertIn("2", command)

    def test_command_to_string(self):
        text = command_to_string(["python", "x y.py"])
        self.assertIn("python", text)
        self.assertIn("'x y.py'", text)

    def test_run_pipeline_writes_dry_run_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "run.json"
            results = run_pipeline(
                {
                    "name": "dry",
                    "log_file": str(log_file),
                    "steps": [{"name": "one", "script": "scripts/x.py", "args": {"flag": True}}],
                },
                dry_run=True,
            )
            self.assertEqual(results[0]["status"], "dry_run")
            self.assertTrue(log_file.exists())
            self.assertIn('"dry"', log_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
