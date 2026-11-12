import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from scripts.build_release import build_release, collect_release_files
from scripts.verify_release import verify_archive, VerificationError
from scripts.privacy_check import inspect, scan

ROOT = Path(__file__).resolve().parents[1]

class PublicationTests(unittest.TestCase):
    def test_reproducible_source_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "project"
            source.mkdir()
            for name, path in collect_release_files(ROOT):
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            (source / "runs").mkdir()
            (source / "runs" / "secret.txt").write_text("local experiment; must be excluded", encoding="utf-8")
            (source / "private").mkdir()
            (source / "private" / "raw.json").write_text("[]", encoding="utf-8")
            one, two = root / "one.zip", root / "two.zip"
            build_release(source, one, source_date_epoch=1788134400)
            build_release(source, two, source_date_epoch=1788134400)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            verified = verify_archive(one)
            names = verified["member_names"]
            self.assertIn("README.md", names)
            self.assertFalse(any(name.startswith(("runs/", "private/")) for name in names))
            original = hashlib.sha256(one.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                build_release(source, one)
            self.assertEqual(original, hashlib.sha256(one.read_bytes()).hexdigest())
            altered = root / "altered.zip"
            with zipfile.ZipFile(one) as before, zipfile.ZipFile(altered, "w") as after:
                for info in before.infolist():
                    data = before.read(info.filename)
                    after.writestr(info, data + b"changed" if info.filename == "README.md" else data)
            with self.assertRaises(VerificationError):
                verify_archive(altered)

    def test_privacy_rules(self):
        self.assertTrue(inspect("private/raw.json", b"[]"))
        self.assertTrue(inspect("weight.safetensors", b"x"))
        self.assertTrue(scan("sk-" + "a" * 30))
        self.assertTrue(scan("/" + "Users/" + "example-person/file.py"))
        self.assertFalse(scan("reader@example.com"))
