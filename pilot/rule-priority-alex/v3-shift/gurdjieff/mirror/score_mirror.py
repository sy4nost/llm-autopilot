"""Frozen scoring for the mirror test (PLAN v2).  python3 -I score_mirror.py out_main.jsonl"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
R4 = lambda v: round(float(v), 4)


def ckey(x):
    if x["fam"] == "R":
        m = re.search(r"\d+\s*(?:[-+*x×/]|mod)\s*\d+", x["user"])
        return "R:" + (re.sub(r"\s+", "", m.group(0)) if m else x["id"])
    return x["cluster"]


items = [x for x in json.load(open(os.path.join(HERE, "..", "items_g.json")))["items"] if x["pool"] == "own" and x["fam"] in ("R", "D")]
lamp = json.load(open(os.path.join(HERE, "lamp_scores.json")))
sc = {}
for l in open(sys.argv[1]):
    r = json.loads(l); sc[(r["id"], r["line"])] = r["sums"]
items = [x for x in items if all((x["id"], k) in sc for k in ("NONE", "MIRROR", "HINT", "GENERIC"))]
N = len(items)
FULL = N == 1399
print(f"items scored with all 4 lines: {N}")
pop = np.array([x["pop"].replace("D/agree", "D/no-conflict").replace("D/plain", "D/no-conflict") for x in items])
fam = np.array([x["fam"] for x in items]); cell = np.array([x["cell"] for x in items]); y = np.array([x["y"] for x in items])
clu = np.array([ckey(x) for x in items])
ok = {k: np.array([int(np.argmax(sc[(x["id"], k)]) == x["key"]) for x in items], float) for k in ("NONE", "MIRROR", "HINT", "GENERIC")}
fires = np.array([lamp[x["id"]]["fires"] for x in items])

print("## Sanity: NONE pick == saved DIRECT pick (exact ties excluded), need >= 98% per family")
sane = True
for f in ("R", "D"):
    ii = [i for i in range(N) if fam[i] == f and (lambda s: s[0] - s[1])(sorted(items[i]["saved_direct"], reverse=True)) >= 1e-3]
    a = np.mean([np.argmax(sc[(items[i]["id"], "NONE")]) == np.argmax(items[i]["saved_direct"]) for i in ii])
    sane &= a >= 0.98; print(f"  {f}: {a:.1%} of {len(ii)}")
sane &= FULL
print("SANITY", "PASS" if sane else "FAIL -> ABORT: no verdict" + ("" if FULL else " (not all 1399 items scored)"))

# placements: weight w = share of the line given to the item (1/0, or expected value for RANDOM-S)
K = {f: int(fires[fam == f].sum()) for f in ("R", "D")}
W = {"BASE": np.zeros(N), "MIRROR": fires.astype(float), "ALL": np.ones(N)}
rs = np.zeros(N)
for p in set(pop):
    t = pop == p; rs[t] = fires[t].mean()
W["RANDOM-S"] = rs
rng = np.random.default_rng(0); jit = rng.uniform(0, 1e-6, N)
margin = np.array([(lambda s: s[0] - s[1])(sorted(sc[(x["id"], "NONE")], reverse=True)) for x in items])
w = np.zeros(N)  # TEMPLATE: whole cells by flip rate, boundary cell gets the remaining share as an expected value
for f in ("R", "D"):
    cs = sorted({c for c in cell[fam == f]}, key=lambda c: (-y[cell == c].mean(), c)); left = K[f]
    for c in cs:
        t = cell == c; n = int(t.sum())
        if left <= 0:
            break
        w[t] = min(1.0, left / n); left -= n
W["TEMPLATE"] = w
for name, key in (("CONF", -margin),):
    w = np.zeros(N)
    for f in ("R", "D"):
        ii = np.where(fam == f)[0]; top = ii[np.argsort(-(key[ii] + jit[ii]))[:K[f]]]; w[top] = 1
    W[name] = w
LINE = {"BASE": "MIRROR", "MIRROR": "MIRROR", "ALL": "MIRROR", "RANDOM-S": "MIRROR", "TEMPLATE": "MIRROR", "CONF": "MIRROR",
        "HINTL": "HINT", "GENERIC": "GENERIC"}
W["HINTL"] = fires.astype(float); W["GENERIC"] = fires.astype(float)
acc_i = {a: W[a] * ok[LINE[a]] + (1 - W[a]) * ok["NONE"] for a in W}

TRAPS = ["R/traps", "D/traps"]; HARM = ["R/cfright", "R/agree", "R/compute", "D/no-conflict"]
P = {p: np.where(pop == p)[0] for p in TRAPS + HARM}


def score(a, pops, idx=None):
    idx = idx or P
    return float(np.mean([acc_i[a][idx[p]].mean() for p in pops]))


def boot(fn):
    out = []
    for scheme in ("item", "cluster"):
        v = []
        for _ in range(B):
            idx = {}
            for p, ii in P.items():
                if scheme == "item":
                    idx[p] = rng.choice(ii, len(ii))
                else:
                    cs = {}
                    for i in ii:
                        cs.setdefault(clu[i], []).append(i)
                    ks = list(cs); idx[p] = np.array([i for k in rng.integers(0, len(ks), len(ks)) for i in cs[ks[k]]])
            v.append(fn(idx))
        out.append((np.percentile(v, 2.5), np.percentile(v, 97.5)))
    return min(o[0] for o in out), max(o[1] for o in out)


print("\n## Lamp firings per population (lamp fires on 30% of each family)")
for p in TRAPS + HARM:
    print(f"  {p:14} {int(fires[P[p]].sum()):4} of {len(P[p])}")
print("\n## Accuracy per population (right answers of 100)")
arms = ["BASE", "MIRROR", "HINTL", "GENERIC", "RANDOM-S", "TEMPLATE", "CONF", "ALL"]
print(f"  {'population':14} " + " ".join(f"{a:>9}" for a in arms))
for p in TRAPS + HARM:
    print(f"  {p:14} " + " ".join(f"{100 * acc_i[a][P[p]].mean():9.1f}" for a in arms))
print(f"  {'traps (mean)':14} " + " ".join(f"{100 * score(a, TRAPS):9.1f}" for a in arms))


def diff(a, b, pops):
    d = score(a, pops) - score(b, pops); lo, hi = boot(lambda idx: score(a, pops, idx) - score(b, pops, idx)); return d, lo, hi


d, lo, hi = diff("MIRROR", "BASE", TRAPS); m1 = R4(d) > 0 and R4(lo) > 0
print(f"\nM1 MIRROR - BASE traps {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need > 0, low > 0 -> {'PASS' if m1 else 'FAIL'}")
best = max(["RANDOM-S", "TEMPLATE", "CONF"], key=lambda a: score(a, TRAPS))
d = score("MIRROR", TRAPS) - score(best, TRAPS)
lo, hi = boot(lambda idx: score("MIRROR", TRAPS, idx) - max(score(a, TRAPS, idx) for a in ("RANDOM-S", "TEMPLATE", "CONF")))
m2 = R4(d) >= 0.03 and R4(lo) > 0
print(f"M2 MIRROR - best simple placement ({best}) traps {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need >= +3, low > 0 -> {'PASS' if m2 else 'FAIL'}")
d, lo, hi = diff("MIRROR", "HINTL", TRAPS); m3 = R4(d) >= 0.03 and R4(lo) > 0
print(f"M3 MIRROR - HINTL traps {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need >= +3, low > 0 -> {'PASS' if m3 else 'FAIL'}"
      + ("  (|diff| < 3: the hint's content does the work, sensor framing adds nothing)" if abs(d) < 0.03 else ""))
d, lo, hi = diff("MIRROR", "ALL", TRAPS + HARM); m5 = R4(d) >= 0
print(f"M5 (6 populations, equal weight) MIRROR - ALL traps+harm {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need >= 0 (point) -> {'PASS' if m5 else 'FAIL'}")
mh = True
for p in HARM:
    d, lo, hi = diff("MIRROR", "BASE", [p]); da, la, ha = diff("ALL", "BASE", [p]); mh &= R4(lo) >= -0.05
    print(f"   harm {p:14} MIRROR-BASE {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]   ALL-BASE {100 * da:+.1f} [{100 * la:+.1f}, {100 * ha:+.1f}]")
print(f"HARM every population MIRROR - BASE low >= -5 -> {'PASS' if mh else 'FAIL'}")
v = sane and m1 and m2 and m3 and m5 and mh
print(f"\nVERDICT: {'ABORT: no verdict (sanity)' if not sane else 'the mirror helps (self-regulation shown)' if v else 'mirror NOT shown'}")
d, lo, hi = diff("HINTL", "BASE", TRAPS)
print(f"report: HINTL - BASE traps {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}];  GENERIC - BASE traps {100 * (score('GENERIC', TRAPS) - score('BASE', TRAPS)):+.1f}")
for c in ("R/cfright/T0", "R/traps/T7"):
    t = cell == c
    print(f"report: {c} n {t.sum()} fires {int(fires[t].sum())}  " + " ".join(f"{a} {100 * acc_i[a][t].mean():.0f}" for a in ("BASE", "MIRROR", "HINTL", "ALL")))
