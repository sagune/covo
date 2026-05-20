import tempfile
import unittest
import zipfile
from pathlib import Path

from covo.archives import unpack_zip_member


class ArchivesTest(unittest.TestCase):
    def test_unpack_zip_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "x.zip"
            output_path = Path(tmp) / "x.txt"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("x.txt", "hello")

            result = unpack_zip_member(zip_path, output_path)

            self.assertEqual(result["status"], "written")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "hello")


if __name__ == "__main__":
    unittest.main()
