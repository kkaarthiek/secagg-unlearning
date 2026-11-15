"""Small helpers: printing, csv files, averages, and where files go."""
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SECUNLEARN_DATA", ROOT / "data"))
STARTED = time.time()


def say(*parts):
    print(" ".join(str(p) for p in parts), flush=True)


def minutes(seconds):
    return f"{seconds / 60:.1f} min"


def norm(vector):
    return float(np.linalg.norm(vector))


def write_csv(path, rows, fields):
    """Write a list of dicts as a csv file with the given columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(pattern_dir, pattern):
    """Every row of every csv in a folder that matches the pattern (values stay text)."""
    rows = []
    for path in sorted(Path(pattern_dir).glob(pattern)):
        with open(path, newline="") as handle:
            rows += list(csv.DictReader(handle))
    return rows


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)
