import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generate_reproducible_data_verify_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_reproducible_data.py", "--verify"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "s1_train.npz" in result.stdout
    assert "s3_test.npz" in result.stdout
