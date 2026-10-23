"""Reviewer pre-check: simulate the PLAN's disguise module on N (change_0) from EXISTING greedy answers (seed 20261020, 30 sets).
Native answer vs fictional-twin answer mapped back by label position; module = native if equal else twin.
python3 -I sim_disguise_N.py <v3-shift dir>"""
import json, os, re, sys, random
V3 = sys.argv[1]
fl = lambda t, L: ([w.upper() for w in re.findall(r"[A-Za-z]+", t) if w.upper() in L] or [None])[0]
def load(sets, res):
    items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(sets))}
    out = {}
    for a in json.load(open(res))["answers"]:
        k = (a["set"], a["arm"], a["probe_ids"][0])
        if k in items and a["arm"] == "change_0":
            it = items[k]; g = fl(a["greedy"], it["labels"])
            out[(a["set"], k[2])] = (None if g is None else it["labels"].index(g), it["labels"].index(it["probes"][0]["key"]),
                                     it["labels"].index(it["probes"][0]["old"]))
    return out
N = load(os.path.join(V3, "native", "sets_native.json"), os.path.join(V3, "native", "results", "Qwen2.5-14B-Instruct.json"))
F = load(os.path.join(V3, "match", "sets_match.json"), os.path.join(V3, "match", "results", "Qwen2.5-14B-Instruct.json"))
assert all(N[k][1:] == F[k][1:] for k in N), "key/old positions differ"
rows = {p: [] for p in ("D1", "D2", "D3", "D4")}
for (s, p), (na, key, old) in N.items():
    fa = F[(s, p)][0]
    mod = na if na == fa else fa
    rows[p].append((s, na == key, fa == key, mod == key, na != fa))
tot = []
for p, r in rows.items():
    n = len(r)
    print(f"{p}: n {n}  normal {sum(x[1] for x in r)/n:5.0%}  twin {sum(x[2] for x in r)/n:5.0%}  module {sum(x[3] for x in r)/n:5.0%}  disagree {sum(x[4] for x in r)/n:5.0%}")
    tot += r
by_set = {}
for s, a, b, c, d in tot:
    by_set.setdefault(s, []).append(c - a)
d = [sum(v) / len(v) for v in by_set.values()]
rng = random.Random(1)
bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
print(f"pooled D1-D4: normal {sum(x[1] for x in tot)/len(tot):.1%} module {sum(x[3] for x in tot)/len(tot):.1%} "
      f"gain {sum(d)/len(d):+.1%} [{bs[50]:+.1%}, {bs[1949]:+.1%}] (set bootstrap)")
