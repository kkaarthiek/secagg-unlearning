import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for folder in ("main", "ablation"):
    sys.path.insert(0, str(ROOT / folder))
