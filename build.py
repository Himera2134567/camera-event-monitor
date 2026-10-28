"""Run in an environment with requirements-build.txt installed."""
import subprocess
import sys
import os
import importlib.util
import shutil
from pathlib import Path
os.environ["YOLO_AUTOINSTALL"] = "false"
if not (3, 12, 10) <= sys.version_info[:3] < (3, 13, 0):
    raise SystemExit("Use Python 3.12.10 or newer from the 3.12 series to build this application.")
root = Path(__file__).resolve().parent
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "CameraEventMonitor", "--collect-all", "streamlit", "--collect-all", "ultralytics", "--copy-metadata", "torch", "--copy-metadata", "ultralytics", "--copy-metadata", "streamlit"]
# Desktop COM, notebooks, training dashboards and HF datasets are not used here.
for unused in ("PySide6", "PyQt5", "matplotlib.tests", "pythoncom", "pywintypes", "win32com", "IPython", "datasets", "tensorboard", "mlxtend"):
    command.extend(["--exclude-module", unused])
command.extend(["--add-data", f"{root / 'app.py'};.", "--hidden-import", "settings", "--hidden-import", "security", str(root / "run.py")])
subprocess.run(command, cwd=root, check=True)
# New torchvision builds use _C_stable/image_stable names that older hooks miss.
vision = Path(importlib.util.find_spec("torchvision").origin).parent
destination = root / "dist" / "CameraEventMonitor" / "_internal" / "torchvision"
for pattern in ("*.pyd", "*.dll"):
    for binary in vision.rglob(pattern):
        target = destination / binary.relative_to(vision)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, target)
