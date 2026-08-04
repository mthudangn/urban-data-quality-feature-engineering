from pathlib import Path
import subprocess
import sys

import pandas as pd

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "run_pipeline.py")], cwd=root, check=True)

expected = [
    root / "results" / "delivery_dirty_cleaned.csv",
    root / "results" / "delivery_missing_imputed.csv",
    root / "results" / "delivery_outlier_cleaned.csv",
    root / "results" / "melbourne_suburb_transformed.csv",
]
for path in expected:
    if not path.exists():
        raise RuntimeError(f"Missing output: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Output is empty: {path}")
print("Smoke test passed")
