"""CICM's own calibration/confirmation split (R9C-SPLIT, 20% calibration per factorial cell), copied from their stale_binding_natural_routing.py."""
import hashlib
from collections import defaultdict


def split(rows, frac=0.2):
    g = defaultdict(list)
    for r in rows:
        c = r["factorial_cell"]; g[(c["same_slot_stale_distance_bin"], c["recent_other_slot_distance_bin"])].append(r)
    cal, conf = [], []
    for k in sorted(g):
        o = sorted(g[k], key=lambda r: hashlib.sha256(f"R9C-SPLIT|{r['id']}".encode()).digest())
        n = max(1, int(round(len(o) * frac))); cal += o[:n]; conf += o[n:]
    return cal, conf
