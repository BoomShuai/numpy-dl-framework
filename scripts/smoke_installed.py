"""Run installed CLI from an unrelated working directory in isolated Python."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    def call(*args):
        result = subprocess.run([sys.executable, "-I", "-m", "numpy_dl", *args], cwd=root,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return json.loads(result.stdout.decode("utf-8"))
    report = call("check")
    assert all(value["passed"] for value in report.values())
    report = call("train", "--epochs", "3", "--checkpoint", "demo.npz")
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    assert (root / "demo.npz").is_file()
    assert call("benchmark")["max_absolute_difference"] < 1e-10
print("PASS: installed CLI works outside checkout")
