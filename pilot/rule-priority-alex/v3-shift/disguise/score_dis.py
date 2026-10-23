"""Frozen scoring for the disguise / shake test (PLAN.md v2).  python3 -I score_dis.py out_main_main.jsonl ALPHA
Answer = option with the largest summed log-prob (report: first-token and mean-per-token for N).
Filter: normal option mass >= 0.5 (item kept or dropped for all methods together).
CIs: 98.75% (Bonferroni over 4 methods), paired bootstrap, 2000 replicates."""
import json, math, sys
import numpy as np

B, LO, HI = 2000, 0.625, 99.375
rows = [json.loads(l) for l in open(sys.argv[1])]
checks = rows[-1].get("checks"); rows = [r for r in rows if "res" in r]
ALPHA = float(sys.argv[2])
print("checks:", checks, "(alpha0_max_diff and repeat_max_diff must be 0)")


def pick(sc, how="sum"):
    v = [sum(s) if how == "sum" else s[0] if how == "first" else sum(s) / len(s) for s in sc]
    return int(np.argmax(v))


def mass(sc):
    return float(sum(math.exp(sum(s)) for s in sc))


kept = [r for r in rows if mass(r["res"]["normal"]) >= 0.5]
print(f"kept {len(kept)} of {len(rows)} after the option-mass filter")
for r in kept:
    r["ok"] = {k: float(pick(v) == r["key"]) for k, v in r["res"].items()}
    r["ans"] = {k: pick(v) for k, v in r["res"].items()}
FAM = {f: [r for r in kept if r["family"] == f] for f in ("N-traps", "N-agree", "N-confirm", "R-traps", "R-same", "R-agree", "C-rule")}

# habit check
n_norm = np.mean([r["ok"]["normal"] for r in FAM["N-traps"]]) if FAM["N-traps"] else float("nan")
N_HABIT = n_norm < 0.85
R_T = sorted({r["cell"] for r in FAM["R-traps"]})
R_HAB = [t for t in R_T if np.mean([r["ok"]["normal"] for r in FAM["R-traps"] if r["cell"] == t]) < 0.85]
R_HABIT = len(R_HAB) > 0
print(f"habit present: N-traps normal right {n_norm:.0%} -> {'yes' if N_HABIT else 'NO (<85% rule)'}; "
      f"R templates with habit (normal right < 85%): {R_HAB}")


def plan(fam):
    """resampling units: N/C by set (grp); R-traps/agree by item within template; R-same by distinct prompt within template"""
    rs = FAM[fam]
    if fam[0] in "NC":
        g = {}
        for k, r in enumerate(rs):
            g.setdefault(r["grp"], []).append(k)
        return ("group", [np.array(v) for v in g.values()])
    strata = {}
    for k, r in enumerate(rs):
        if fam == "R-traps" and r["cell"] not in R_HAB:
            continue
        strata.setdefault(r["cell"], {}).setdefault(r["grp"] if fam == "R-same" else k, []).append(k)
    return ("strat", {t: [np.array(v) for v in d.values()] for t, d in strata.items()})


def stat(fam, idx, f):
    """f(rows subset) -> value; R families: equal-weight mean over templates"""
    rs = FAM[fam]
    if isinstance(idx, dict):
        vals = [f([rs[k] for k in ix]) for t, ix in idx.items() if len(ix)]
        return float(np.mean(vals)) if vals else float("nan")
    return f([rs[k] for k in idx])


def draw(fam, rng):
    kind, p = plan(fam)
    if kind == "group":
        return np.concatenate([p[j] for j in rng.integers(0, len(p), len(p))]) if p else np.array([], int)
    return {t: np.concatenate([u[j] for j in rng.integers(0, len(u), len(u))]) for t, u in p.items()}


def full(fam):
    kind, p = plan(fam)
    return np.concatenate(p) if kind == "group" and p else {t: np.concatenate(u) for t, u in p.items()} if kind == "strat" else np.array([], int)


acc = lambda m: (lambda rs: float(np.mean([r["ok"][m] for r in rs])) if rs else float("nan"))
diff = lambda m, base="normal": (lambda rs: float(np.mean([r["ok"][m] - r["ok"][base] for r in rs])) if rs else float("nan"))


def randflip(m):
    def f(rs):
        if not rs:
            return float("nan")
        fl = np.mean([r["ans"][m] != r["ans"]["normal"] for r in rs])
        e = [(1 - fl) * r["ok"]["normal"] + fl * (1 - r["ok"]["normal"]) / (len(r["cands"]) - 1) for r in rs]
        return float(np.mean([r["ok"][m] for r in rs]) - np.mean(e))
    return f


def ci(fam, f, seed=0):
    rng = np.random.default_rng(seed); xs = [stat(fam, draw(fam, rng), f) for _ in range(B)]
    return stat(fam, full(fam), f), np.nanpercentile(xs, LO), np.nanpercentile(xs, HI)


def ci2(f, seed=0):
    """traps of N and R pooled with weight 0.5 each (only families with a habit), resampled separately"""
    fams = [x for x, h in (("N-traps", N_HABIT), ("R-traps", R_HABIT)) if h]
    rng = np.random.default_rng(seed)
    pt = np.mean([stat(x, full(x), f) for x in fams])
    xs = [np.mean([stat(x, draw(x, rng), f) for x in fams]) for _ in range(B)]
    return pt, np.nanpercentile(xs, LO), np.nanpercentile(xs, HI)


print("\n## Right answers (%), normal and every variant (R: equal-weight mean over templates; R-traps over habit templates)")
VAR = ["normal", "disguise", "bang", "wake", "placebo", "instr"] + [f"noise_{a}" for a in (0.05, 0.1, 0.2, 0.4)]
print(f"{'family':11}{'n':>5}" + "".join(f"{v:>11}" for v in VAR))
for fam in FAM:
    if not FAM[fam]:
        continue
    line = f"{fam:11}{len(FAM[fam]):>5}"
    for v in VAR:
        line += f"{stat(fam, full(fam), acc(v)) * 100:>11.1f}" if all(v in r['ok'] for r in FAM[fam]) else f"{'-':>11}"
    print(line)

METHODS = ["disguise", "bang", "wake", f"noise_{ALPHA}"]
print(f"\n######## PRE-REGISTERED VERDICT (each method alone; 98.75% CIs; noise alpha = {ALPHA})")
for m in METHODS:
    print(f"\n### {m}")
    ok, notes = True, []
    fam_ok = {}
    for fam, hab in (("N-traps", N_HABIT), ("R-traps", R_HABIT)):
        if not hab:
            notes.append(f"{fam}: no habit, not scored"); continue
        if not all(m in r["ok"] for r in FAM[fam]):
            notes.append(f"{fam}: method not applicable"); fam_ok[fam] = False; continue
        e, lo, hi = ci(fam, diff(m))
        g = e >= 0.10 and lo > 0
        if fam == "R-traps":
            pos = sum(stat(fam, {t: full(fam)[t]}, diff(m)) > 0 for t in R_HAB)
            g = g and pos >= math.ceil(len(R_HAB) / 2)
            print(f"  {fam} gain {e:+.1%} [{lo:+.1%}, {hi:+.1%}], positive in {pos}/{len(R_HAB)} habit templates -> {'PASS' if g else 'FAIL'}")
        else:
            print(f"  {fam} gain {e:+.1%} [{lo:+.1%}, {hi:+.1%}] -> {'PASS' if g else 'FAIL'}")
        fam_ok[fam] = g
    for fam in ("N-agree", "N-confirm", "R-same", "R-agree"):
        if not FAM[fam] or not all(m in r["ok"] for r in FAM[fam]):
            continue
        e, lo, hi = ci(fam, diff(m))
        h = e >= -0.05 and lo >= -0.10; ok &= h
        print(f"  harm {fam:10} {e:+.1%} [{lo:+.1%}, {hi:+.1%}] need >= -5% and low >= -10% -> {'PASS' if h else 'FAIL'}")
    if m in ("bang", "wake"):
        for base in ("placebo", "instr"):
            e, lo, hi = ci2(diff(m, base))
            c = e >= 0.05 and lo > 0; ok &= c
            print(f"  vs {base} on traps (N,R 0.5 each): {e:+.1%} [{lo:+.1%}, {hi:+.1%}] need >= +5% and low > 0 -> {'PASS' if c else 'FAIL'}")
    if m.startswith("noise"):
        e, lo, hi = ci2(randflip(m))
        c = e >= 0.05 and lo > 0; ok &= c
        print(f"  vs random flips at the same rate (traps): {e:+.1%} [{lo:+.1%}, {hi:+.1%}] need >= +5% and low > 0 -> {'PASS' if c else 'FAIL'}")
    scored = [v for v in fam_ok.values()]
    if not scored:
        v = "NOTHING TO SAY (no habit in either family)"
    elif not ok or not all(scored):
        v = "FAIL"
    elif len(scored) == 1:
        v = "PARTIAL (passes on the only family with a habit)"
    else:
        v = "PASS"
    print(f"  {'; '.join(notes)}{' | ' if notes else ''}VERDICT {m}: {v}")

print("\n######## Report only")
for fam in ("N-traps", "N-agree"):
    for c in sorted({r["cell"] for r in FAM[fam]}):
        rs = [r for r in FAM[fam] if r["cell"] == c]
        print(f"  {fam} {c}: n {len(rs)} " + " ".join(f"{v} {acc(v)(rs):.0%}" for v in VAR if all(v in r['ok'] for r in rs)))
for c in R_T:
    rs = [r for r in FAM["R-traps"] if r["cell"] == c]
    print(f"  R-traps {c}: n {len(rs)} " + " ".join(f"{v} {acc(v)(rs):.0%}" for v in VAR if all(v in r['ok'] for r in rs)))
rs = FAM["N-traps"] + FAM["N-agree"]
for how in ("first", "mean"):
    print(f"  N disguise scored by {how}: traps {np.mean([pick(r['res']['disguise'], how) == r['key'] for r in FAM['N-traps']]):.0%}"
          f" agree {np.mean([pick(r['res']['disguise'], how) == r['key'] for r in FAM['N-agree']]):.0%} (normal by {how}: "
          f"{np.mean([pick(r['res']['normal'], how) == r['key'] for r in FAM['N-traps']]):.0%} / {np.mean([pick(r['res']['normal'], how) == r['key'] for r in FAM['N-agree']]):.0%})")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())) if len(p) and len(n) else float("nan")


for fam in ("N-traps", "R-traps"):
    rs = [r for r in FAM[fam] if "disguise" in r["ans"] and r["old"] is not None]
    flag = [float(r["ans"]["disguise"] != r["ans"]["normal"]) for r in rs]
    y = [float(r["ans"]["normal"] == r["old"]) for r in rs]
    agree_ok = [r["ok"]["normal"] for r, f in zip(rs, flag) if not f]
    print(f"  disagreement as a detector of habit errors, {fam}: AUC {auc(flag, y):.3f}; flagged {np.mean(flag):.0%}; "
          f"normal accuracy where not flagged {np.mean(agree_ok) if agree_ok else float('nan'):.0%}")
if FAM["C-rule"]:
    print("  C-rule (in-context copy habit): " + " ".join(f"{v} {acc(v)(FAM['C-rule']):.0%}" for v in VAR if all(v in r['ok'] for r in FAM['C-rule'])))
