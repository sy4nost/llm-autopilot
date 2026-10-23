"""Frozen scoring for think-aloud on independent tasks (PLAN.md v2).  python3 -I score_ti.py out_main.jsonl
DIRECT / INSTR answers = forced choice. THINK / THINKI answer = option parsed from the final "Answer:" line,
falling back to the forced choice after "Answer:" when nothing parses."""
import json, re, sys
import numpy as np

B = 2000
R4 = lambda v: round(v, 4)  # thresholds compared after rounding (float edge cases)
rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", rows[-1].get("checks")); rows = [r for r in rows if "res" in r]
MAIN = ["memo", "neqa", "hind", "unknown", "alg"]
CTRL = ["memo", "neqa", "hind", "alg"]
REP = ["modus", "quote", "pattern", "redef"]
WORDS = {"memo", "quote", "pattern"}
norm = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace("−", "-"))


def pick(sc, how="sum"):
    return int(np.argmax([sum(s) if how == "sum" else sum(s) / len(s) for s in sc]))


def parse(r, g):
    """option named on the final "Answer:" line, else None (-> forced choice). PLAN v2.1"""
    if "Answer:" not in g:
        return None
    line = g[g.rindex("Answer:") + 7:].strip().split("\n")[0].strip()
    cands = r["cands"]
    if r["task"] in ("alg", "redef"):  # last number on the line
        nums = re.findall(r"-?\d+", line)
        return cands.index(nums[-1]) if nums and nums[-1] in cands else None
    if r["task"] in WORDS:  # final word of the line only
        toks = norm(line)
        cn = [norm(c)[-1] if norm(c) else c.lower() for c in cands]
        return cn.index(toks[-1]) if toks and toks[-1] in cn and cn.count(toks[-1]) == 1 else None
    if all(re.fullmatch(r"[A-D]", c) for c in cands):  # standalone capital letter, first one
        m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", line)
        return cands.index(m.group(1)) if m and m.group(1) in cands else None
    yn = {"y": "y", "yes": "y", "n": "n", "no": "n"}  # Y/N and Yes/No
    cy = [yn.get(c.lower()) for c in cands]
    for t in norm(line):
        if t in yn and yn[t] in cy:
            return cy.index(yn[t])
    return None


for r in rows:
    a = {"direct": pick(r["res"]["direct"]), "direct_mean": pick(r["res"]["direct"], "mean"),
         "rawc": pick(r["res"]["rawc"]), "rawc_mean": pick(r["res"]["rawc"], "mean"), "instr": pick(r["res"]["instr"])}
    for t in ("think", "thinki"):
        f = pick(r["res"][t]); p = parse(r, r["gen_" + t])
        a[t] = f if p is None else p; a[t + "_forced"] = f; r["parsed_" + t] = p is not None
    r["ans"] = a; r["ok"] = {k: v == r["key"] for k, v in a.items()}
    v = sorted((sum(s) for s in r["res"]["direct"]), reverse=True); r["margin"] = v[0] - v[1]
by = {}
for r in rows:
    by.setdefault((r["task"], r["pop"]), []).append(r)
DIRECTS = ["direct", "direct_mean", "rawc"]


def acc(rs, a):
    return float(np.mean([r["ok"][a] for r in rs])) if rs else float("nan")


def boot(stat, keys, seed=0):
    """stat(dict key->rows) -> float; item scheme and cluster scheme, wider 95% interval"""
    rng = np.random.default_rng(seed); pt = stat({k: by[k] for k in keys}); xi, xc = [], []
    cl = {k: {} for k in keys}
    for k in keys:
        for r in by[k]:
            cl[k].setdefault(r["cluster"], []).append(r)
    for _ in range(B):
        xi.append(stat({k: [by[k][i] for i in rng.integers(0, len(by[k]), len(by[k]))] for k in keys}))
        d = {}
        for k in keys:
            ks = list(cl[k]); d[k] = [r for j in rng.integers(0, len(ks), len(ks)) for r in cl[k][ks[j]]]
        xc.append(stat(d))
    return pt, min(np.percentile(xi, 2.5), np.percentile(xc, 2.5)), max(np.percentile(xi, 97.5), np.percentile(xc, 97.5))


def delta(a, b):
    return lambda d: float(np.mean([acc(d[k], a) - acc(d[k], b) for k in d]))


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    if len(p) < 5 or len(n) < 5:
        return float("nan")
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))


fmt = lambda x: f"{x[0] * 100:+.1f} [{x[1] * 100:+.1f}, {x[2] * 100:+.1f}]"
ARMS = ["direct", "direct_mean", "rawc", "rawc_mean", "instr", "think", "thinki", "think_forced", "thinki_forced"]
print("\n## Share right per task / population")
print(f"  {'task':8} {'pop':7} {'n':>4} " + " ".join(f"{a[:11]:>11}" for a in ARMS) + "   parsed T/TI  agree T/TI  trunc T/TI")
for t in MAIN + REP + ["mmlu"]:
    for pop in ("trap", "control"):
        rs = by.get((t, pop), [])
        if rs:
            print(f"  {t:8} {pop:7} {len(rs):4} " + " ".join(f"{acc(rs, a):11.1%}" for a in ARMS) +
                  f"   {np.mean([r['parsed_think'] for r in rs]):4.0%} {np.mean([r['parsed_thinki'] for r in rs]):4.0%}"
                  f"  {np.mean([r['ans']['think'] == r['ans']['think_forced'] for r in rs]):4.0%} {np.mean([r['ans']['thinki'] == r['ans']['thinki_forced'] for r in rs]):4.0%}"
                  f"  {np.mean([r['trunc_think'] for r in rs]):4.0%} {np.mean([r['trunc_thinki'] for r in rs]):4.0%}")
print("  per class, main traps (shows class shifts):")
for t in MAIN:
    for c in (0, 1):
        rs = [r for r in by[(t, "trap")] if r["key"] == c]
        print(f"    {t:8} right={c} n {len(rs)} " + " ".join(f"{a} {acc(rs, a):.0%}" for a in ("direct", "instr", "think", "thinki")))

TR = [(t, "trap") for t in MAIN]; CT = [(t, "control") for t in CTRL]
print("\n######## PRE-REGISTERED VERDICT (main traps: memo, NeQA, hindsight, into-the-unknown, repetitive-algebra; equal weight)")
verd = {}
for arm, name in (("think", "C1 THINK "), ("thinki", "C2 THINKI")):
    ok = True
    for dv in DIRECTS:
        x = boot(delta(arm, dv), TR); p = R4(x[0]) >= 0.05 and R4(x[1]) > 0; ok &= p
        print(f"{name} - {dv:11} {fmt(x)} need >= +5, low > 0 -> {'PASS' if p else 'FAIL'}")
    npos = sum(R4(acc(by[k], arm) - acc(by[k], "direct")) > 0 for k in TR); ok &= npos >= 3
    print(f"{name} families with a gain over DIRECT: {npos} of 5 (need >= 3) -> {'PASS' if npos >= 3 else 'FAIL'}")
    hc = boot(delta(arm, "direct"), CT); pf = [R4(acc(by[k], arm) - acc(by[k], "direct")) for k in CT]
    pc = R4(hc[0]) >= -0.05 and R4(hc[1]) >= -0.10 and min(pf) >= -0.10
    hm = boot(delta(arm, "direct"), [("mmlu", "trap")]); pm = R4(hm[0]) >= -0.02 and R4(hm[1]) >= -0.05
    print(f"C3 {name.split()[1]} harm on controls (habit answer right) {fmt(hc)} need >= -5, low >= -10, every family >= -10 (per family {[round(v * 100, 1) for v in pf]}) -> {'PASS' if pc else 'FAIL'}")
    print(f"C3 {name.split()[1]} harm on MMLU (1000) {fmt(hm)} need >= -2, low >= -5 -> {'PASS' if pm else 'FAIL'}")
    verd[arm] = ok and pc and pm
print(f"VERDICT THINK : {'HELPS on independent tasks' if verd['think'] else 'NOT SHOWN'}")
print(f"VERDICT THINKI: {'HELPS on independent tasks' if verd['thinki'] else 'NOT SHOWN'}")

print("\n######## Report only")
for a, b in (("think", "instr"), ("thinki", "instr"), ("thinki", "think"), ("instr", "direct")):
    print(f"  {a.upper():6} - {b.upper():6} main traps {fmt(boot(delta(a, b), TR))} | controls {fmt(boot(delta(a, b), CT))}")
print("  per task deltas vs DIRECT [95%]  (THINK | THINKI | INSTR):")
for k in TR + CT + [(t, "trap") for t in REP] + [("mmlu", "trap")]:
    if k in by:
        print(f"    {k[0]:8} {k[1]:7} " + " | ".join(fmt(boot(delta(a, "direct"), [k])) for a in ("think", "thinki", "instr")))
print("  detector 'THINK answer != DIRECT answer' -> DIRECT wrong (per task, traps+controls, needs >= 5 of each):")
for t in MAIN:
    rs = by[(t, "trap")] + by.get((t, "control"), []); y = [not r["ok"]["direct"] for r in rs]
    print(f"    {t:8} AUC flag {auc([r['ans']['think'] != r['ans']['direct'] for r in rs], y):.2f}  low-confidence {auc([-r['margin'] for r in rs], y):.2f}  DIRECT errors {sum(y)}")
mm = by[("mmlu", "trap")]
print(f"    MMLU flag rate {np.mean([r['ans']['think'] != r['ans']['direct'] for r in mm]):.1%}")
