"""Machine gates on the smoke run (must all pass before the main run).  python3 check_smoke.py smoke"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1] if len(sys.argv) > 1 else "smoke"
rows = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "data", "cicm_natural_factorial_otherdist_l0.jsonl")))}
out = json.load(open(os.path.join(HERE, f"out_{tag}.json"))); ok = True


def gate(name, cond, info):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}: {info}")


gate("run finished", os.path.exists(os.path.join(HERE, f"out_{tag}.DONE")), f"{len(out)} rows")
cells = {str(rows[o["id"]]["factorial_cell"]) for o in out}
gate("all 6 cells present", len(cells) == 6, f"{len(cells)} cells")
chk = [o for o in out if "final_full" in o]
d1 = max(abs(a - b) for o in chk for a, b in zip(o["final"], o["final_full"]))
d2 = max(abs(a - b) for o in chk for a, b in zip(o["off_top20"], o["off_full"]))
d3 = max(abs(a - b) for o in chk for a, b in zip(o["final"], o["ident"]))
gate("cached = full recompute, heads on", d1 <= 0.1, f"max diff {d1:.4f} nat on {len(chk)} rows")
gate("cached = full recompute, top20 off", d2 <= 0.1, f"max diff {d2:.4f}")
gate("hook active, nothing zeroed = normal", d3 <= 0.1, f"max diff {d3:.4f}")
d4 = max(abs(a - b) for o in chk for a, b in zip(o["final"], o["off_top20"]))
gate("heads-off actually changes scores", d4 > 0.1, f"max diff {d4:.4f}")
lab = []
for o in out:
    r = rows[o["id"]]; v = o["cands"][int(np.argmax(o["final"]))]
    lab.append("correct" if v == r["current_value"] else "stale" if v in r["stale_values"] else "other")
st = lab.count("stale") / len(lab)
gate("stale answers exist (>=25%)", st >= 0.25, f"correct {lab.count('correct')}, stale {lab.count('stale')}, other {lab.count('other')}")
ag = n = 0
for o in out:
    hits = [v for v in o["cands"] if v.lower() in o["gen"].lower()]
    if len(hits) == 1:
        n += 1; ag += hits[0] == o["cands"][int(np.argmax(o["final"]))]
gate("forced pick = value in greedy answer (>=85%)", n and ag / n >= 0.85, f"{ag}/{n}")
print("ALL GATES PASS" if ok else "GATES FAILED - do not run main")
