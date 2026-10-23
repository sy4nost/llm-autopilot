"""v2 check: how many (arm x probe) cells would reach >=10 habit errors and >=10 non-errors at the v2 sizes,
using per-cell greedy/argmax rates from earlier runs (match/results greedy; newtasks2 argmax of 'normal').
Also binomial probability that a cell qualifies, given the observed rate. Reads result files only."""
import json, sys, math, collections, numpy as np
def qual_prob(n, p, k=10):  # P(X>=k and n-X>=k), X~Bin(n,p)
    return sum(math.comb(n, x) * p**x * (1-p)**(n-x) for x in range(k, n-k+1))
out = []
# ---- C-rule from match (30 sets) -> 90 sets
M = json.load(open("match/results/Qwen2.5-14B-Instruct.json"))["answers"]
S = {(x["set"], x["arm"]): None for x in []}
items = json.load(open("match/sets_match.json"))
key = {}
for it in items:
    pr = it["probes"][0]; key[(it["set"], it["arm"], pr["id"])] = (pr["key"], pr["old"])
cell = collections.defaultdict(lambda: [0, 0])
for a in M:
    if a["arm"] not in ("A_change_20", "A_change_60", "B_change_60", "C_change_60"): continue
    pid = a["probe_ids"][0]; k, o = key[(a["set"], a["arm"], pid)]
    cell[(a["arm"], pid)][0] += 1; cell[(a["arm"], pid)][1] += int(a["greedy"].strip().strip(".").upper() == o.upper() and o != k)
print("C-rule (earlier: 30 sets, greedy). v2: 90 per cell")
tq = 0
for c in sorted(cell):
    n, e = cell[c]; p = e / n; q = qual_prob(90, p); tq += q
    print(f"  {c[0]:12s} {c[1]}  rate {p:.2f}  expected errors/90 {90*p:5.1f}  P(cell qualifies) {q:.2f}")
print(f"  expected qualifying cells: {tq:.1f} of {len(cell)}")
# ---- parcels / bonus from newtasks2 (20 sets) -> 80 sets
R = [json.loads(l) for l in open("newtasks2/new2_Qwen2.5-14B-Instruct.jsonl")][1:]
for task in ("parcels", "bonus"):
    cell = collections.defaultdict(lambda: [0, 0])
    for r in R:
        if r["task"] != task or r["arm"] not in ("A60", "E_nocopy"): continue
        ans = r["labels"][int(np.argmax(r["scores"]["normal"]))]
        cell[(r["arm"], r["probe"])][0] += 1; cell[(r["arm"], r["probe"])][1] += int(ans == r["old"] and r["old"] != r["key"])
    print(f"{task} (earlier: 20 sets, argmax). v2: 80 per cell")
    tq = 0
    for c in sorted(cell):
        n, e = cell[c]; p = e / n; q = qual_prob(80, p); tq += q
        print(f"  {c[0]:9s} {c[1]}  rate {p:.2f} ({e}/{n})  expected errors/80 {80*p:5.1f}  P(qualifies) {q:.2f}")
    print(f"  expected qualifying cells: {tq:.1f} of {len(cell)}")
