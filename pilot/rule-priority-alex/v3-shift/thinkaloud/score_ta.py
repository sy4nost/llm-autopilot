"""Frozen scoring for the think-aloud test (PLAN.md v2 + v2.1).  python3 -I score_ta.py out_main.jsonl
Forced choice for every arm (largest summed log-prob), no mass filter. CIs 95%: both bootstrap schemes, wider one used."""
import json, math, re, sys
import numpy as np

B = 2000
rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", rows[-1].get("checks")); rows = [r for r in rows if "res" in r]
by = {r["uid"]: r for r in rows}
pick = lambda sc: int(np.argmax([sum(s) for s in sc]))
STRONG = ["in reality", "real world", "real-world", "actually", "normally", "usually", "in fact", "conventional", "typically",
          "commonly", "well-known", "well known", "premise", "contradict", "differs from", "approximately 3.14", "2.718", "standard"]
GENERIC = {"actually", "standard", "typically", "commonly"}
WEAK = ["however", "but"]
PROMPT = {}
sets = json.load(open(sys.argv[1].replace("out_main.jsonl", "sets_ta.json").replace("out_smoke.jsonl", "sets_smoke.json")))["items"]
for it in sets:
    PROMPT[it["uid"]] = it["think"].lower()


def lamp(r, words):
    g, p = r["gen_think"].lower(), PROMPT[r["uid"]]
    return any(len(re.findall(re.escape(w), g)) > len(re.findall(re.escape(w), p)) for w in words)


def parse(g, cands):
    if "Answer:" not in g:
        return None
    a = g[g.rindex("Answer:") + 7:].strip().strip("*").strip().strip('"\'').rstrip(".").strip().lower()
    for i, c in enumerate(cands):
        if a == c.lower() or (re.fullmatch(r"-?\d+", a) and re.fullmatch(r"-?\d+", c) and int(a) == int(c)):
            return i
    return None


for r in rows:
    r["ans"] = {k: pick(v) for k, v in r["res"].items()}
    r["ok"] = {k: float(v == r["key"]) for k, v in r["ans"].items()}
    v = sorted(sum(s) for s in r["res"]["direct"]); r["margin"] = v[-1] - v[-2]
    r["L"] = lamp(r, STRONG); r["L2"] = lamp(r, [w for w in STRONG if w not in GENERIC]); r["LW"] = lamp(r, WEAK)
    r["parsed"] = parse(r["gen_think"], r["cands"])
FAMS = ["R", "N", "K", "D"]
CONTROLS = [("R", "cfright"), ("R", "agree"), ("R", "compute"), ("N", "agree"), ("N", "confirm"), ("K", "agree"), ("K", "plain"),
            ("D", "agree"), ("D", "plain")]


def unit(r):
    return r["grp"] if r["family"] == "N" else r["cell"] if r["family"] in "KD" else None


def resample(rs, rng, cluster):
    """cluster: N by set, K/D by hook then item, R by item within template (twins follow their trap); item: items within (family,pop)"""
    out = []
    if cluster:
        groups = {}
        for r in rs:
            if r["family"] == "R":
                groups.setdefault(("R", r["pop"], r["cell"]), []).append([r])
            elif r["family"] == "N":
                groups.setdefault(("N", r["pop"]), {}).setdefault(r["grp"], []).append(r)
            else:
                groups.setdefault((r["family"], r["pop"]), {}).setdefault(r["cell"], []).append(r)
        for k, g in groups.items():
            if isinstance(g, list):
                for j in rng.integers(0, len(g), len(g)):
                    out += g[j]
            else:
                ks = list(g)
                for j in rng.integers(0, len(ks), len(ks)):
                    x = g[ks[j]]
                    out += [x[i] for i in rng.integers(0, len(x), len(x))] if k[0] != "N" else x
    else:
        groups = {}
        for r in rs:
            groups.setdefault((r["family"], r["pop"]), []).append(r)
        for g in groups.values():
            out += [g[i] for i in rng.integers(0, len(g), len(g))]
    return out


def boot(rs, stat, seed=0):
    pt = stat(rs); xs = {0: [], 1: []}
    for c in (0, 1):
        rng = np.random.default_rng(seed + c)
        for _ in range(B):
            xs[c].append(stat(resample(rs, rng, bool(c))))
    lo = min(np.nanpercentile(xs[0], 2.5), np.nanpercentile(xs[1], 2.5)); hi = max(np.nanpercentile(xs[0], 97.5), np.nanpercentile(xs[1], 97.5))
    return pt, lo, hi


def fam_gain(rs, f, a, b, adjust):
    tr = [r for r in rs if r["family"] == f and r["pop"] == "traps"]
    if not tr:
        return float("nan")
    if f != "R":
        return float(np.mean([r["ok"][a] - r["ok"][b] for r in tr]))
    per = []
    for t in sorted({r["cell"] for r in tr}):
        d = []
        for r in tr:
            if r["cell"] != t:
                continue
            x = r["ok"][a] - r["ok"][b]
            if adjust and "twin" in r:
                tw = by[r["twin"]]; x -= tw["ok"][a] - tw["ok"][b]
            d.append(x)
        per.append(np.mean(d))
    return float(np.mean(per))


def gain(rs, a, b, adjust):
    return float(np.nanmean([fam_gain(rs, f, a, b, adjust) for f in FAMS]))


TR = [r for r in rows if r["pop"] == "traps"]
print("\n## Right answers (%) per family and population, all arms (forced choice)")
ARMS = ["direct", "instr", "think", "thinki", "disg"]
for f in FAMS:
    for pop in sorted({r["pop"] for r in rows if r["family"] == f}):
        rs = [r for r in rows if r["family"] == f and r["pop"] == pop]
        print(f"  {f} {pop:8} n {len(rs):4} " + " ".join(f"{a} {100 * np.mean([r['ok'][a] for r in rs]):5.1f}" for a in ARMS if all(a in r['ok'] for r in rs))
              + f" | lamp {np.mean([r['L'] for r in rs]):.0%} trunc {np.mean([r['trunc_think'] for r in rs]):.0%} unparsed {np.mean([r['parsed'] is None for r in rs]):.0%}")
print("  R traps per template: " + " | ".join(f"{t} direct {np.mean([r['ok']['direct'] for r in TR if r['cell'] == t]):.0%} think {np.mean([r['ok']['think'] for r in TR if r['cell'] == t]):.0%} instr {np.mean([r['ok']['instr'] for r in TR if r['cell'] == t]):.0%}" for t in sorted({r['cell'] for r in TR if r['family'] == 'R'})))

print("\n######## PRE-REGISTERED VERDICT")
c1 = boot(TR, lambda s: gain(s, "think", "direct", True))
p1 = c1[0] >= 0.10 and c1[1] > 0
print(f"1. THINK - DIRECT on traps (R compute-adjusted, families equal): {c1[0]:+.1%} [{c1[1]:+.1%}, {c1[2]:+.1%}] need >= +10, low > 0 -> {'PASS' if p1 else 'FAIL'}")
c2 = boot(TR, lambda s: gain(s, "think", "instr", False))
p2 = c2[0] >= 0.05 and c2[1] > 0
print(f"2. THINK - INSTR on traps: {c2[0]:+.1%} [{c2[1]:+.1%}, {c2[2]:+.1%}] need >= +5, low > 0 -> {'PASS' if p2 else 'FAIL'}")
p3 = True
for f, pop in CONTROLS:
    rs = [r for r in rows if r["family"] == f and r["pop"] == pop]
    if not rs:
        print(f"3. harm {f} {pop}: no items"); continue
    e = boot(rs, lambda s: float(np.mean([r["ok"]["think"] - r["ok"]["direct"] for r in s])))
    h = e[0] >= -0.05 and e[1] >= -0.10; p3 &= h
    print(f"3. harm {f} {pop:8} {e[0]:+.1%} [{e[1]:+.1%}, {e[2]:+.1%}] need >= -5, low >= -10 -> {'PASS' if h else 'FAIL'}")
print(f"VERDICT 'thinking helps' (1 & 2 & 3): {'PASS' if p1 and p2 and p3 else 'FAIL'}")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))


RIVALS = {"margin": lambda r: -r["margin"], "think!=direct": lambda r: float(r["ans"]["think"] != r["ans"]["direct"]),
          "instr!=direct": lambda r: float(r["ans"]["instr"] != r["ans"]["direct"]),
          "disg!=direct": lambda r: float(r["ans"]["disg"] != r["ans"]["direct"]) if "disg" in r["ans"] else None,
          "S2 no": lambda r: float(r["ans"]["truth"] == 1) if "truth" in r["ans"] else None}


def cells(rs):
    c = {}
    for r in rs:
        if r["pop"] == "traps" and r["ans"]["direct"] in (r["key"], r["habit"]):
            c.setdefault((r["family"], r["cell"]), []).append(r)
    return {k: v for k, v in c.items() if sum(r["ans"]["direct"] == r["habit"] for r in v) >= 5 and sum(r["ans"]["direct"] == r["key"] for r in v) >= 5}


def cell_auc(rs, f):
    v = []
    for x in cells(rs).values():
        s = [f(r) for r in x]
        if any(z is None for z in s):
            continue
        v.append(auc(s, [r["ans"]["direct"] == r["habit"] for r in x]))
    return float(np.mean(v)) if v else float("nan")


C = cells(TR)
print(f"\n4. measurable cells (>= 5 habit errors and >= 5 right in DIRECT): {len(C)} -> {sorted(C)}")
if len(C) < 6:
    print("   fewer than 6 cells -> criterion 4 NOT MEASURABLE; VERDICT 'verbal lamp': INCONCLUSIVE")
else:
    L = boot(TR, lambda s: cell_auc(s, lambda r: float(r["L"])))
    best = lambda s: max(cell_auc(s, f) for f in RIVALS.values())
    D4 = boot(TR, lambda s: cell_auc(s, lambda r: float(r["L"])) - best(s))
    for k, f in RIVALS.items():
        print(f"   rival {k:14} within-cell AUC {cell_auc(TR, f):.3f}")
    p4 = L[0] >= 0.70 and L[1] > 0.55 and D4[0] >= 0.05 and D4[1] > 0
    print(f"   lamp within-cell AUC {L[0]:.3f} [{L[1]:.3f}, {L[2]:.3f}]; minus best rival {D4[0]:+.3f} [{D4[1]:+.3f}, {D4[2]:+.3f}] "
          f"need >= 0.70 low > 0.55 and >= +0.05 low > 0 -> {'PASS' if p4 else 'FAIL'}")
    print(f"VERDICT 'verbal lamp': {'PASS' if p4 else 'FAIL'}")
    print(f"   sensitivity: lamp without {sorted(GENERIC)}: within-cell AUC {cell_auc(TR, lambda r: float(r['L2'])):.3f}")

print("\n######## Report only")
print(f"  1 unadjusted: {gain(TR, 'think', 'direct', False):+.1%}; without truncated THINK items: "
      f"{gain([r for r in TR if not r['trunc_think']], 'think', 'direct', True):+.1%}")
for f in FAMS:
    print(f"  family {f}: THINK-DIRECT {fam_gain(TR, f, 'think', 'direct', True):+.1%} (unadj {fam_gain(TR, f, 'think', 'direct', False):+.1%}), "
          f"THINK-INSTR {fam_gain(TR, f, 'think', 'instr', False):+.1%}, THINKI-DIRECT {fam_gain(TR, f, 'thinki', 'direct', False):+.1%}")
ag = [r for r in rows if r["parsed"] is not None]
print(f"  parsed answer agrees with forced THINK answer: {np.mean([r['parsed'] == r['ans']['think'] for r in ag]):.0%} of {len(ag)} parsed "
      f"({len(rows) - len(ag)} unparsed); parsed accuracy on traps {np.mean([r['parsed'] == r['key'] for r in TR if r['parsed'] is not None]):.0%}")
for f in FAMS:
    tr = [r for r in TR if r["family"] == f]
    on = [r for r in tr if r["L"]]; off = [r for r in tr if not r["L"]]
    print(f"  {f} traps: lamp on {len(on)} (THINK right {np.mean([r['ok']['think'] for r in on]) if on else float('nan'):.0%}, DIRECT habit "
          f"{np.mean([r['ans']['direct'] == r['habit'] for r in on]) if on else float('nan'):.0%}) | off {len(off)} (THINK right "
          f"{np.mean([r['ok']['think'] for r in off]) if off else float('nan'):.0%}, DIRECT habit {np.mean([r['ans']['direct'] == r['habit'] for r in off]) if off else float('nan'):.0%}) | weak-word rate {np.mean([r['LW'] for r in tr]):.0%}")
