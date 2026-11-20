"""Run the whole project end to end:  python run_all.py
Downloads the data if needed, cleans it, runs EDA, trains/evaluates every model, writes
figures to reports/figures and tables to reports/tables.  Fixed seed -> identical results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ["scripts/01_prepare_eda.py", "scripts/02_train_evaluate.py"]:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)
print("\nAll done. See reports/figures and reports/tables.")
