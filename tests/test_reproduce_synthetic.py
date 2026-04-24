import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reproduce_synthetic_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_synthetic.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--step" in result.stdout
    assert "canonical synthetic artifacts" in result.stdout
