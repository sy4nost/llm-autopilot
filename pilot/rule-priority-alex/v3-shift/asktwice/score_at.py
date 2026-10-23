"""Frozen scoring for the ask-twice checker (PLAN.md v2 + v2.1).  python3 -I score_at.py out_main_main.jsonl dev_choice.json
Answer = forced choice: option with the largest summed log-prob, no option-mass filter (PLAN v3.1). Hooks whose plain
right < 80% (main) are dropped from every population. Family counts only if entered on dev AND main traps right < 85%."""
import json, math, sys
import numpy as np

B = 2000
rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", rows[-1].get("checks")); rows = [r for r in rows if "res" in r]
ch = json.load(open(sys.argv[2]))
POPS = {"U": ["traps", "agree", "irrelevant"], "K": ["traps", "agree", "knowledge", "irrelevant"],
        "D": ["traps", "agree", "irrelevant"]}
mass = lambda sc: sum(math.exp(sum(s)) for s in sc)


def pick(sc, how="sum"):
    return int(np.argmax([sum(s) if how == "sum" else sum(s) / len(s) for s in sc]))


def share(sc, k):
    v = np.array([sum(s) for s in sc]); p = np.exp(v - v.max()); return float(p[k] / p.sum())


print(f"{len(rows)} items, forced choice (no option-mass filter); share with normal option mass < 0.5: "
      f"{np.mean([mass(r['res']['normal']) < 0.5 for r in rows]):.0%}")


def annotate(r, how="sum"):
    a = pick(r["res"]["normal"], how); r["ans"] = a; r["wrong"] = a != r["key"]
    v = sorted(sum(s) for s in r["res"]["normal"]); r["margin"] = v[-1] - v[-2]
    r["fl"] = dict(AT=pick(r["res"]["disg"], how) != a, S=r["pop"] != "plain",
                   S2=bool(r["res"].get("truth")) and sum(r["res"]["truth"][1]) > sum(r["res"]["truth"][0]),
                   P=pick(r["res"]["para"], how) != a, I=pick(r["res"]["instr"], how) != a, M=r["margin"] < ch["M_thresh"],
                   O=mass(r["res"]["normal"]) < 0.5)
    r["fl"]["S3"] = r["fl"]["S2"] and r["hook_in_q"]
    r["fl"]["H"] = r["fl"]["S3"] and r["cell"] in ch["H_cells"].get(r["family"], [])
    r["s_AT"] = share(r["res"]["normal"], a) - share(r["res"]["disg"], a); r["s_M"] = -r["margin"]


for r in rows:
    annotate(r)
drop = set()
for f in POPS:
    for c in {r["cell"] for r in rows if r["family"] == f}:
        pl = [r for r in rows if r["family"] == f and r["cell"] == c and r["pop"] == "plain"]
        if pl and np.mean([not r["wrong"] for r in pl]) < 0.80:
            drop.add((f, c))
print("hooks dropped (plain right < 80%):", sorted(drop))
rows = [r for r in rows if (r["family"], r["cell"]) not in drop]
FAM = []
for f in POPS:
    tr = [r for r in rows if r["family"] == f and r["pop"] == "traps"]
    if f not in ch["variants"] or not tr:
        print(f"family {f}: not entered on dev"); continue
    right = np.mean([not r["wrong"] for r in tr]); hab = sum(r["ans"] == r["habit"] for r in tr)
    print(f"family {f} (variant {ch['variants'][f]}): main traps right {right:.0%} (dev {[t['traps_right'] for t in ch['table'] if t['family'] == f and t['variant'] == ch['variants'][f]][0]:.0%}), "
          f"habit choices {hab}, other wrong {sum(r['wrong'] for r in tr) - hab} -> {'counts' if right < 0.85 else 'no habit, excluded'}")
    if right < 0.85:
        FAM.append(f)
R = [r for r in rows if r["family"] in FAM and r["pop"] in POPS[r["family"]]]
RULES = ["AT", "S", "S2", "S3", "M", "P", "I", "H", "O"]


def weights(rs):
    w = np.zeros(len(rs)); cnt = {}
    for r in rs:
        cnt[(r["family"], r["pop"])] = cnt.get((r["family"], r["pop"]), 0) + 1
    for i, r in enumerate(rs):
        w[i] = 1.0 / (len(FAM) * len(POPS[r["family"]]) * cnt[(r["family"], r["pop"])])
    return w


def BA(rs, rule):
    if not rs:
        return float("nan")
    w = weights(rs); y = np.array([r["wrong"] for r in rs]); fl = np.array([r["fl"][rule] for r in rs])
    c = (w * fl)[y].sum() / w[y].sum() if y.any() else float("nan")
    fa = (w * fl)[~y].sum() / w[~y].sum() if (~y).any() else float("nan")
    return 0.5 * (c + 1 - fa)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))


def cell_auc(rs, key):
    cells = {}
    for r in rs:
        if r["pop"] == "traps":
            cells.setdefault((r["family"], r["cell"]), []).append(r)
    v = [auc([r[key] for r in x], [r["wrong"] for r in x]) for x in cells.values()
         if sum(r["wrong"] for r in x) >= 3 and sum(not r["wrong"] for r in x) >= 3]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def boot(stat, seed=0):
    """both schemes; returns (point, low, high) using the wider interval"""
    rng = np.random.default_rng(seed); pt = stat(R)
    byfc = {}
    for r in R:
        byfc.setdefault(r["family"], {}).setdefault(r["cell"], []).append(r)
    byfp = {}
    for r in R:
        byfp.setdefault((r["family"], r["pop"]), []).append(r)
    xs_c, xs_i = [], []
    for _ in range(B):
        s = []
        for f, cells in byfc.items():
            ks = list(cells)
            for j in rng.integers(0, len(ks), len(ks)):
                x = cells[ks[j]]; s += [x[i] for i in rng.integers(0, len(x), len(x))]
        xs_c.append(stat(s))
        s = []
        for x in byfp.values():
            s += [x[i] for i in rng.integers(0, len(x), len(x))]
        xs_i.append(stat(s))
    lo = min(np.nanpercentile(xs_c, 2.5), np.nanpercentile(xs_i, 2.5)); hi = max(np.nanpercentile(xs_c, 97.5), np.nanpercentile(xs_i, 97.5))
    return pt, lo, hi


print(f"\nfamilies counted: {FAM}")
if len(FAM) < 2:
    print("VERDICT: NOTHING TO SAY (fewer than 2 families with a habit)"); sys.exit()

print("\n## Flag rates per population (share flagged; 'err' = share of normal answers wrong)")
for f in FAM:
    for pop in POPS[f]:
        rs = [r for r in R if r["family"] == f and r["pop"] == pop]
        print(f"  {f} {pop:10} n {len(rs):4} err {np.mean([r['wrong'] for r in rs]):5.0%} " +
              " ".join(f"{k} {np.mean([r['fl'][k] for r in rs]):4.0%}" for k in RULES))
print("\n## Balanced accuracy on the mix")
ba = {k: boot(lambda s, k=k: BA(s, k)) for k in RULES}
for k in RULES:
    print(f"  {k:3} {ba[k][0]:.3f} [{ba[k][1]:.3f}, {ba[k][2]:.3f}]")

print("\n######## PRE-REGISTERED VERDICT")
d1 = boot(lambda s: BA(s, "AT") - max(BA(s, "S2"), BA(s, "S3")))
c1 = d1[0] >= 0.05 and d1[1] > 0
print(f"1. BA(AT) - max(BA(S2), BA(S3)) = {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] need >= +0.05, low > 0 -> {'PASS' if c1 else 'FAIL'}")
d2 = boot(lambda s: BA(s, "AT") - max(BA(s, k) for k in ("M", "P", "I", "H", "O")))
c2 = d2[0] >= 0.05 and d2[1] > 0
print(f"2. BA(AT) - max(BA(M), BA(P), BA(I), BA(H), BA(O)) = {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}] need >= +0.05, low > 0 -> {'PASS' if c2 else 'FAIL'}")
a_at, ncell = cell_auc(R, "s_AT")
if ncell == 0:
    c3 = None; print("3. within-cell AUC: not measurable (no trap cell with >= 3 errors and >= 3 right)")
else:
    A = boot(lambda s: cell_auc(s, "s_AT")[0]); DM = boot(lambda s: cell_auc(s, "s_AT")[0] - cell_auc(s, "s_M")[0])
    c3 = A[0] >= 0.75 and A[1] > 0.6 and DM[0] >= 0.10 and DM[1] > 0
    print(f"3. within-cell AUC(AT) {A[0]:.3f} [{A[1]:.3f}, {A[2]:.3f}] over {ncell} cells; minus AUC(M) {DM[0]:+.3f} [{DM[1]:+.3f}, {DM[2]:+.3f}] "
          f"need >= 0.75 low > 0.6 and >= +0.10 low > 0 -> {'PASS' if c3 else 'FAIL'}")
if c1 and c2 and c3:
    v = "PASS"
elif c1 and c2 and c3 is None:
    v = "INCONCLUSIVE (criteria 1-2 pass, within-cell AUC not measurable)"
else:
    v = "FAIL"
print(f"VERDICT ask-twice: {v}")

print("\n######## Report only")
irr = [r for r in R if r["pop"] == "irrelevant"]
print(f"  4. AT false alarms on irrelevant: {np.mean([r['fl']['AT'] for r in irr]):.1%} (silent by construction when hook not in question)")
for f in FAM:
    rs = [r for r in R if r["family"] == f]
    print(f"  family {f}: BA " + " ".join(f"{k} {BA(rs, k):.3f}" for k in RULES) + f" | within-cell AUC AT {cell_auc(rs, 's_AT')[0]:.3f} M {cell_auc(rs, 's_M')[0]:.3f} ({cell_auc(rs, 's_AT')[1]} cells)")
for f in FAM:
    for nz in sorted({r["nonce"] for r in R if r["family"] == f}):
        tr = [r for r in R if r["family"] == f and r["nonce"] == nz and r["pop"] == "traps"]
        kn = [r for r in R if r["family"] == f and r["nonce"] == nz and r["pop"] == "knowledge"]
        print(f"  nonce {f}/{nz}: traps n {len(tr)} err {np.mean([r['wrong'] for r in tr]):.0%} AT flag {np.mean([r['fl']['AT'] for r in tr]):.0%}"
              + (f" | knowledge AT flag {np.mean([r['fl']['AT'] for r in kn]):.0%}" if kn else ""))
print("  accuracy of the normal answer where AT does not flag, per population:")
for f in FAM:
    for pop in POPS[f]:
        rs = [r for r in R if r["family"] == f and r["pop"] == pop and not r["fl"]["AT"]]
        print(f"    {f} {pop:10} n {len(rs):4} right {np.mean([not r['wrong'] for r in rs]) if rs else float('nan'):.0%}")
print("  answer policies (share right): normal / always-disguise / always-instr")
for f in FAM:
    for pop in POPS[f]:
        rs = [r for r in R if r["family"] == f and r["pop"] == pop]
        print(f"    {f} {pop:10} {np.mean([not r['wrong'] for r in rs]):.0%} / {np.mean([pick(r['res']['disg']) == r['key'] for r in rs]):.0%} / "
              f"{np.mean([pick(r['res']['instr']) == r['key'] for r in rs]):.0%}")
R2 = [dict(r) for r in R]
for r in R2:
    annotate(r, "mean")
print("  length-normalised scoring check, BA: " + " ".join(f"{k} {BA(R2, k):.3f}" for k in RULES))
