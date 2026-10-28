"""Launch the bundled Streamlit application locally."""
import json
import os
import sys
import webbrowser
from pathlib import Path
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

def main():
    if "--self-test" in sys.argv:
        import cv2
        import numpy
        import streamlit
        import torch
        from ultralytics import YOLO
        from security import hash_password, verify_password
        from settings import MODEL_PATH
        assert verify_password("demo-check", hash_password("demo-check"))
        model = YOLO(str(MODEL_PATH))
        predictions = model(numpy.zeros((64, 64, 3), dtype=numpy.uint8), imgsz=64, device="cpu", verbose=False)
        assert len(predictions) == 1
        result = {"ok": True, "python": sys.version, "torch": torch.__version__, "streamlit": streamlit.__version__, "cuda_required": False, "original_model_cpu_inference": "passed on synthetic frame"}
        if "--self-test-report" in sys.argv:
            Path(sys.argv[sys.argv.index("--self-test-report")+1]).write_text(json.dumps(result), encoding="utf-8")
        if sys.stdout:
            print(json.dumps(result))
        return
    from streamlit.web import cli
    app = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "app.py"
    port = os.environ.get("CAMERA_MONITOR_PORT", "8501")
    headless = os.environ.get("CAMERA_MONITOR_HEADLESS", "false")
    sys.argv = ["streamlit", "run", str(app), "--server.address=127.0.0.1", f"--server.port={port}", f"--server.headless={headless}", "--browser.gatherUsageStats=false", "--global.developmentMode=false"]
    cli.main()

if __name__ == "__main__":
    main()
