"""The report, code and saved d8-v5 results agree (runs report/verify_reproduction.py)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results/d8-v5/artifacts").is_dir(), reason="saved d8-v5 results not present")
def test_report_reproduces_from_saved_results():
    proc = subprocess.run([sys.executable, "report/verify_reproduction.py"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
