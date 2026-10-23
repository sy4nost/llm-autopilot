"""Frozen dev rule (PLAN.md v2 + v2.1).  python3 -I choose_dev.py out_dev_dev2.jsonl dev_choice.json
Per family: variant whose normal right on traps is closest to 50% within [15%, 85%] (ties -> lower variant); the family enters
only if habit-option choices >= 3x distractor choices on that variant's traps and plain right >= 90%.
M threshold: margin cut maximising BA of "flag = margin < t" vs "normal wrong" on dev traps + plain of the chosen variants.
H cells: hooks whose dev trap error rate on the chosen variant >= 50%."""
import json, math, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", rows[-1].get("checks")); rows = [r for r in rows if "res" in r]
tot = lambda s: sum(s)
mass = lambda sc: sum(math.exp(tot(s)) for s in sc)
rows = [r for r in rows if mass(r["res"]["normal"]) >= 0.5]
for r in rows:
    v = sorted([tot(s) for s in r["res"]["normal"]]); r["ans"] = int(np.argmax([tot(s) for s in r["res"]["normal"]]))
    r["wrong"] = r["ans"] != r["key"]; r["margin"] = v[-1] - v[-2]
choice, table = {}, []
for f in "UKD":
    best = None
    for v in sorted({r["variant"] for r in rows if r["family"] == f}):
        tr = [r for r in rows if r["family"] == f and r["variant"] == v and r["pop"] == "traps"]
        pl = [r for r in rows if r["family"] == f and r["variant"] == v and r["pop"] == "plain"]
        right = np.mean([not r["wrong"] for r in tr]); hab = sum(r["ans"] == r["habit"] for r in tr)
        dis = sum(r["wrong"] and r["ans"] != r["habit"] for r in tr); pr = np.mean([not r["wrong"] for r in pl])
        ok = 0.15 <= right <= 0.85 and hab >= 3 * dis and pr >= 0.90
        table.append(dict(family=f, variant=v, n_traps=len(tr), traps_right=round(float(right), 3), habit=int(hab), distractor=int(dis),
                          plain_right=round(float(pr), 3), n_plain=len(pl), eligible=bool(ok)))
        print(f"{f} v{v}: traps n {len(tr)} right {right:.0%} habit {hab} distractor {dis} | plain n {len(pl)} right {pr:.0%} -> {'eligible' if ok else '-'}")
        if ok and (best is None or abs(right - 0.5) < best[0] - 1e-12):
            best = (abs(right - 0.5), v)
    if best:
        choice[f] = best[1]
print("chosen variants:", choice)
sel = [r for r in rows if r["family"] in choice and r["variant"] == choice[r["family"]]]
y = np.array([r["wrong"] for r in sel]); m = np.array([r["margin"] for r in sel])
bestt, bestba = None, -1
for t in sorted(set(m)) + [max(m) + 1]:
    fl = m < t
    ba = 0.5 * ((fl[y].mean() if y.any() else 0) + (1 - fl[~y].mean() if (~y).any() else 0))
    if ba > bestba + 1e-12:
        bestt, bestba = float(t), ba
print(f"M threshold {bestt} (dev BA {bestba:.3f})")
H = {}
for f, v in choice.items():
    cells = sorted({r["cell"] for r in sel if r["family"] == f})
    H[f] = [c for c in cells if np.mean([r["wrong"] for r in sel if r["family"] == f and r["cell"] == c and r["pop"] == "traps"]) >= 0.5]
    print(f"H cells {f}: {H[f]}")
json.dump(dict(variants=choice, M_thresh=bestt, H_cells=H, table=table), open(sys.argv[2], "w"), indent=1)
print("families entered:", len(choice), "(need >= 2, else stop: nothing to say)")
