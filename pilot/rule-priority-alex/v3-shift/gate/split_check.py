"""Honesty check: pick lamp+fix settings on sets 15-22 only, score the frozen choice on sets 23-29.
python3 -I split_check.py ../eras/eras_Qwen2.5-14B-Instruct.jsonl ../eras/boost_Qwen2.5-14B-Instruct.jsonl"""
import json, sys, itertools
import numpy as np
E = {(r["set"], r["arm"], r["probe"]): r for r in map(json.loads, list(open(sys.argv[1]))[1:])}
B = {(r["set"], r["arm"], r["probe"]): r for r in map(json.loads, list(open(sys.argv[2]))[1:])}
am = lambda v: int(np.argmax(v))
lead = lambda v, p: (lambda a: a[p] - np.delete(a, p).max())(np.asarray(v, float))
def on(sc, q, d):
    p = am(sc["normal"]); return am(sc[q]) != p or lead(sc[q], p) < (1 - d) * lead(sc["normal"], p)
ARMS = ("A60", "E_nocopy", "E_copy")
def score(sets, q, d, fix):
    ks = [k for k in E if k[0] in sets and k[1] in ARMS]
    right = np.mean([B[k]["labels"][am(B[k]["scores"][fix] if on(E[k]["scores"], q, d) else E[k]["scores"]["normal"])] == E[k]["key"] for k in ks])
    nk = [k for k in E if k[0] in sets and k[1] == "N_same"]
    return right, np.mean([on(E[k]["scores"], q, d) for k in nk])
def per_group(sets, q, d, fix):
    out = {}
    for arm in ARMS:
        ks = [k for k in E if k[0] in sets and k[1] == arm and k[2] in ("D1", "D3")]
        out[arm] = (np.mean([E[k]["labels"][am(E[k]["scores"]["normal"])] == E[k]["key"] for k in ks]),
                    np.mean([B[k]["labels"][am(B[k]["scores"][fix] if on(E[k]["scores"], q, d) else E[k]["scores"]["normal"])] == E[k]["key"] for k in ks]), len(ks))
    return out
fixes = [s for s in next(iter(B.values()))["scores"] if s != "normal"]
grid = list(itertools.product(("w0.3", "w0.1", "w0.03", "w0"), (0.2, 0.35, 0.5, 0.65, 0.8), fixes))
A, T = set(range(15, 23)), set(range(23, 30))
# choose: best right on half A, with lamp firing on <=10% of unchanged-rule cases
best = max((g for g in grid if score(A, *g)[1] <= 0.10), key=lambda g: score(A, *g)[0])
print("chosen on sets 15-22:", best, "| grid size", len(grid))
for name, S in (("pick half 15-22", A), ("unseen half 23-29", T)):
    r, f = score(S, *best); print(f"\n{name}: all D1-D4 right {r:.0%}, lamp on unchanged rule {f:.0%}")
    for arm, (n, g, k) in per_group(S, *best).items(): print(f"  {arm:9} traps (n={k}): normal {n:.0%} -> gated {g:.0%}")
r_all = sorted(score(T, *g)[0] for g in grid if score(T, *g)[1] <= 0.10)
print(f"\non unseen half, chosen setting ranks among {len(r_all)} allowed: median {np.median(r_all):.0%}, best {r_all[-1]:.0%}, chosen {score(T, *best)[0]:.0%}")
