"""Put the repo root on sys.path and force a non-interactive matplotlib backend."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "figures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
