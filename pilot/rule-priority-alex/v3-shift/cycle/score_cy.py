"""Frozen scoring for the life-cycle frame test (PLAN.md).  python3 -I score_cy.py out_main.jsonl
Forced choice (largest summed log-prob). Gains: 4 families equal weight, R templates equal weight, R traps adjusted by their
compute twin (same as thinkaloud/score_ta.py). CIs 95%: item and cluster bootstrap, wider one used."""
import json, os, sys
import numpy as np

B = 2000
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", [r["checks"] for r in rows if "checks" in r]); rows = [r for r in rows if "res" in r]
by = {r["uid"]: r for r in rows}
ARMS = ["base", "death", "goal", "placebo", "facts"]
FAMS = ["R", "N", "K", "D"]
CONTROLS = [("R", "cfright"), ("R", "agree"), ("R", "compute"), ("N", "agree"), ("N", "confirm"), ("K", "agree"), ("K", "plain"),
            ("D", "agree"), ("D", "plain")]
# BASE = thinkaloud DIRECT arm on the same items (reproduced 78/78 by smoke v1); attached here, not re-run
_ta = {}
for l in open(os.path.join(HERE, "..", "thinkaloud", "out_main.jsonl")):
    x = json.loads(l)
    if "res" in x:
        _ta[x["uid"]] = x["res"]
for r in rows:
    r["res"]["base"] = _ta[r["uid"]]["direct"]; r["res"]["instr_user"] = _ta[r["uid"]]["instr"]
for r in rows:
    r["ans"] = {a: int(np.argmax([sum(s) for s in r["res"][a]])) for a in ARMS + ["instr_user"]}
    r["ok"] = {a: float(r["ans"][a] == r["key"]) for a in ARMS + ["instr_user"]}
    r["p"] = {}
    for a in ARMS:
        v = np.array([sum(s) for s in r["res"][a]]); e = np.exp(v - v.max()); r["p"][a] = float(e.max() / e.sum())


def resample(rs, rng, cluster):
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
    return pt, min(np.nanpercentile(xs[0], 2.5), np.nanpercentile(xs[1], 2.5)), max(np.nanpercentile(xs[0], 97.5), np.nanpercentile(xs[1], 97.5))


def fam_gain(rs, f, a, b, adjust=True):
    tr = [r for r in rs if r["family"] == f and r["pop"] == "traps"]
    if not tr:
        return float("nan")
    if f != "R":
        return float(np.mean([r["ok"][a] - r["ok"][b] for r in tr]))
    per = []
    for t in sorted({r["cell"] for r in tr}):
        d = []
        for r in tr:
            if r["cell"] == t:
                x = r["ok"][a] - r["ok"][b]
                if adjust and r.get("twin") in by:
                    tw = by[r["twin"]]; x -= tw["ok"][a] - tw["ok"][b]
                d.append(x)
        per.append(np.mean(d))
    return float(np.mean(per))


def gain(rs, a, b, adjust=True):
    return float(np.nanmean([fam_gain(rs, f, a, b, adjust) for f in FAMS]))


TR = [r for r in rows if r["pop"] == "traps"]
print("\n## Right answers (%) per family and population (forced choice)")
for f in FAMS:
    for pop in sorted({r["pop"] for r in rows if r["family"] == f}):
        rs = [r for r in rows if r["family"] == f and r["pop"] == pop]
        print(f"  {f} {pop:8} n {len(rs):4} " + " ".join(f"{a} {100 * np.mean([r['ok'][a] for r in rs]):5.1f}" for a in ARMS))

print("\n######## PRE-REGISTERED VERDICT")
c1 = boot(TR, lambda s: gain(s, "death", "placebo"))
p1 = c1[0] >= 0.05 and c1[1] > 0
print(f"1. DEATH - PLACEBO on traps: {c1[0]:+.1%} [{c1[1]:+.1%}, {c1[2]:+.1%}] need >= +5, low > 0 -> {'PASS' if p1 else 'FAIL'}"
      + ("   (two-sided reading: CI high < 0 -> THE FRAME HURTS)" if c1[2] < 0 else ""))
cf = boot(TR, lambda s: gain(s, "facts", "placebo"))
print(f"   manipulation check FACTS - PLACEBO on traps: {cf[0]:+.1%} [{cf[1]:+.1%}, {cf[2]:+.1%}] "
      f"(if CI low <= 0, the system-prompt channel moved nothing and a DEATH FAIL is uninformative)")
p2 = True
for f, pop in CONTROLS:
    rs = [r for r in rows if r["family"] == f and r["pop"] == pop]
    if not rs:
        print(f"2. harm {f} {pop}: no items"); continue
    e = boot(rs, lambda s: float(np.mean([r["ok"]["death"] - r["ok"]["placebo"] for r in s])))
    h = e[0] >= -0.05 and e[1] >= -0.10; p2 &= h
    print(f"2. harm {f} {pop:8} {e[0]:+.1%} [{e[1]:+.1%}, {e[2]:+.1%}] need >= -5, low >= -10 -> {'PASS' if h else 'FAIL'}")
print(f"VERDICT 'life-cycle frame with reset helps' (1 & 2): {'PASS' if p1 and p2 else 'FAIL'}")

print("\n######## Secondary (pre-registered, no pass/fail)")
for a, b in [("goal", "placebo"), ("death", "goal"), ("death", "base"), ("placebo", "base"), ("facts", "placebo"), ("death", "facts"),
             ("instr_user", "base")]:
    c = boot(TR, lambda s: gain(s, a, b))
    rkd = boot(TR, lambda s: float(np.nanmean([fam_gain(s, f, a, b) for f in "RKD"])))
    print(f"  {a.upper()} - {b.upper()} on traps: {c[0]:+.1%} [{c[1]:+.1%}, {c[2]:+.1%}]; R/K/D only {rkd[0]:+.1%} [{rkd[1]:+.1%}, {rkd[2]:+.1%}]")
    if (a, b) in [("death", "placebo"), ("goal", "placebo"), ("facts", "placebo")]:
        for f in FAMS:
            e = boot([r for r in TR if r["family"] == f], lambda s: fam_gain(s, f, a, b))
            print(f"      {f}: {e[0]:+.1%} [{e[1]:+.1%}, {e[2]:+.1%}]")
for f in FAMS:
    e = boot([r for r in TR if r["family"] == f], lambda s: fam_gain(s, f, "death", "placebo"))
    print(f"  DEATH - PLACEBO family {f}: {e[0]:+.1%} [{e[1]:+.1%}, {e[2]:+.1%}]")
CTL = [r for r in rows if r["pop"] != "traps"]
for a in ARMS:
    wr = [r for r in TR if not r["ok"][a]]
    print(f"  {a:8} controls right {np.mean([r['ok'][a] for r in CTL]):.1%}; traps: wrong {len(wr)}, mean confidence when wrong "
          f"{np.mean([r['p'][a] for r in wr]) if wr else float('nan'):.3f}, when right {np.mean([r['p'][a] for r in TR if r['ok'][a]]):.3f}")

print("\n######## Report only")
for lo, hi in [(1, 50), (51, 150), (151, 200)]:
    s = [r for r in TR if lo <= r["k"] <= hi]
    print(f"  counter k {lo}-{hi}: traps n {len(s)}, DEATH - PLACEBO raw {np.mean([r['ok']['death'] - r['ok']['placebo'] for r in s]):+.1%}")
print(f"  unadjusted DEATH - PLACEBO: {gain(TR, 'death', 'placebo', False):+.1%}")
