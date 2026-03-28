"""
Запуск Streamlit UI из корня проекта (удобная обёртка).

  python scripts/launch_ui.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_ROOT / "app" / "ui" / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
