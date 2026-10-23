"""Independent audit recompute (does NOT import score_*.py, make_*.py or check_*.py).
python3 -I recompute.py  [V3 dir, default: parent of this file]
1. watch/: re-derives per-example 'stale' truth, key and old with an own text solver, checks they equal sets_watch.json
   and the 'truth' field of the raw jsonl; aligns raw rows to sets items; recomputes every accuracy table and the
   pre-registered verdict with a paired cluster bootstrap over (task,set) written from scratch; adds diagnostics
   (watch==oracle when flags==truth, flag-only dependence, test-copy flags, alternative CIs, near-duplicate items).
2. watch_public/: recomputes accuracies, paired deltas, item bootstrap, flips, class shifts.
"""
import json, os, re, sys, random, collections, hashlib
import numpy as np

V3 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
W = os.path.join(V3, "watch"); P = os.path.join(V3, "watch_public")

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---------------------------------------------------------------- own text solver
EXL = re.compile(r"^Example (\d+): (.+?) -> (\S+)$", re.M)

def parse(prompt):
    head = prompt.split("Below are")[0].split("Answer for the TEST")[0]
    if "ROBOT COMMAND" in head:
        return "robot", {"d": dict(re.findall(r"the word '(\w+)' means ([A-Z]+)", head))}
    if "PARCEL ROUTING" in head:
        T = re.search(r"heavy parcel goes by ([A-Z]+)", head).group(1)
        A = re.search(r"RULE 2:.*goes by ([A-Z]+)", head).group(1)
        S = re.search(r"RULE 3:.*goes by ([A-Z]+)", head).group(1)
        V = re.search(r"RULE 4: All remaining parcels go by ([A-Z]+)", head).group(1)
        r2 = re.search(r"RULE 2:.*?(express|standard) goes by", head).group(1)
        return "parcels", {"lab": (T, A, S, V), "r2": r2}
    if "WITH A BONUS" in head:
        val = {c: int(v) for c, v in re.findall(r"- an? (\w+) token is worth (\d) point", head)}
        m = re.search(r"BONUS RULE: If a pile contains an? (\w+) token and, at the same time, (no|an?) (\w+) token, add (\d+)", head)
        return "bonus", {"val": val, "x": m.group(1), "neg": m.group(2) == "no", "y": m.group(3), "b": int(m.group(4))}
    val = {c: int(v) for c, v in re.findall(r"- an? (\w+) token is worth (\d) point", head)}
    return "points", {"val": val}

def apply_note(kind, R, prompt):
    m = re.search(r"IMPORTANT UPDATE: (.*)\n", prompt)
    R = json.loads(json.dumps(R))
    if not m or "reviewed and stays the same" in m.group(1):
        return R
    t = m.group(1)
    if kind == "robot":
        w, a = re.search(r"the word '(\w+)' means ([A-Z]+)", t).groups(); R["d"][w] = a
    elif kind == "points":
        c, v = re.search(r"an? (\w+) token is worth (\d) point", t).groups(); R["val"][c] = int(v)
    elif kind == "parcels":
        R["r2"] = re.search(r"insured and, at the same time, (express|standard) goes by", t).group(1)
    else:
        R["neg"] = re.search(r"at the same time, (no|an?) \w+ token", t).group(1) == "no"
    return R

def solve(kind, R, case):
    if kind == "robot":
        return R["d"][re.fullmatch(r"command '(\w+)'", case).group(1)]
    if kind == "points":
        return str(sum(int(k) * R["val"][c] for k, c in re.findall(r"(\d) (\w+)", case)))
    if kind == "bonus":
        cols = re.findall(r"1 (\w+)", case)
        b = R["x"] in cols and ((R["y"] not in cols) if R["neg"] else (R["y"] in cols))
        return str(sum(R["val"][c] for c in cols) + R["b"] * b)
    T, A, S, V = R["lab"]
    heavy = case.startswith("a heavy"); abroad = "abroad" in case
    ins = ", insured," in case; exp = "express delivery" in case
    if heavy: return T
    if ins and (exp if R["r2"] == "express" else not exp): return A
    return S if abroad else V

# ---------------------------------------------------------------- watch: data integrity
print("=" * 100, "\nWATCH (synthetic tasks)\n")
sets = json.load(open(os.path.join(W, "sets_watch.json")))
items = sets["items"]
print("seed in sets_watch.json:", sets["seed"], "| items", len(items))
L = [json.loads(l) for l in open(os.path.join(W, "watch_Qwen2.5-14B-Instruct.jsonl"))]
checks, R = L[0]["checks"], L[1:]
print("checks row:", {k: v for k, v in checks.items() if not k.startswith("example")})
assert len(R) == len(items)
bad = collections.Counter()
for it, r in zip(items, R):
    for k in ("task", "set", "arm", "probe", "labels", "key", "old"):
        if it[k] != r[k]: bad["row/item mismatch " + k] += 1
    kind, base = parse(it["prompt"]); cur = apply_note(kind, base, it["prompt"])
    exs = EXL.findall(it["prompt"])
    my_truth = [solve(kind, cur, c) != a for _, c, a in exs]
    if my_truth != [e["stale"] for e in it["examples"]]: bad["truth vs sets"] += 1
    if my_truth != r["truth"]: bad["truth vs jsonl"] += 1
    if len(r["flags"]) != len(exs): bad["flag count"] += 1
    test = re.search(r"TEST CASE: (.+)\n", it["prompt"]).group(1)
    if solve(kind, cur, test) != it["key"]: bad["key"] += 1
    if solve(kind, base, test) != it["old"] and it["arm"] != "F0": bad["old"] += 1
    # examples before the note under the base rule, after under cur
    up = it["prompt"].find("IMPORTANT UPDATE")
    for mm in EXL.finditer(it["prompt"]):
        rule = base if (up < 0 or mm.start() < up) else apply_note(kind, base, it["prompt"])
        if solve(kind, rule, mm.group(2)) != mm.group(3): bad["example not correct at its time"] += 1
print("integrity problems (own solver):", dict(bad) or "none")

# ---------------------------------------------------------------- watch: scoring
am = lambda v: int(np.argmax(v))
ans = lambda r, p: r["labels"][am(r["scores"][p])]
right = lambda r, p: float(ans(r, p) == r["key"])
oldans = lambda r, p: float(ans(r, p) == r["old"] and r["old"] != r["key"])
POLS = ["normal", "watch_b40", "watch_all", "oracle_b40"]
TASKS = ("parcels", "bonus", "robot", "points")
ARMS = ("A60", "E_nocopy", "E_copy", "N60", "R60", "N_same", "F0")
G = lambda tasks, arms, probes: [r for r in R if r["task"] in tasks and r["arm"] in arms and r["probe"][0] in probes]

def cluster_boot(rows, pol, base="normal", B=10000, seed=1):
    """paired cluster bootstrap: resample (task,set) clusters with replacement; statistic = pooled mean of
    per-item (right(pol) - right(base)) over all items in the resampled clusters."""
    cl = collections.defaultdict(list)
    for r in rows: cl[(r["task"], r["set"])].append(right(r, pol) - right(r, base))
    keys = sorted(cl); s = np.array([sum(cl[k]) for k in keys]); n = np.array([len(cl[k]) for k in keys])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), (B, len(keys)))
    xs = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(xs, 2.5), np.percentile(xs, 97.5)

def item_boot(rows, pol, base="normal", B=10000, seed=2):
    d = np.array([right(r, pol) - right(r, base) for r in rows]); rng = np.random.default_rng(seed)
    xs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(xs, 2.5), np.percentile(xs, 97.5)

for task in TASKS:
    rows = [r for r in R if r["task"] == task]
    f = np.array([x for r in rows for x in r["flags"]]); t = np.array([x for r in rows for x in r["truth"]])
    print(f"\n### {task}: stale caught {f[t].mean():.1%} of {t.sum()} | wrongly flagged {f[~t].mean():.2%} of {(~t).sum()}"
          f" | precision {t[f].mean():.1%}")
    print(f"{'group':22}{'n':>4}{'lamp':>6}" + "".join(f"{p:>11}" for p in POLS) + f"{'old(n)':>8}{'old(w)':>8}")
    for arm in ARMS:
        for pr, nm in (("T", "trap"), ("S", "untouched")):
            g = G((task,), (arm,), (pr,))
            print(f"{arm + ' ' + nm:22}{len(g):>4}{np.mean([any(r['flags']) for r in g]):>6.0%}"
                  + "".join(f"{100*np.mean([right(r, p) for r in g]):>11.1f}" for p in POLS)
                  + f"{np.mean([oldans(r,'normal') for r in g]):>8.0%}{np.mean([oldans(r,'watch_b40') for r in g]):>8.0%}")

print("\n### PRE-REGISTERED VERDICT, recomputed (watch_b40 - normal; cluster bootstrap over 20 sets, B=10000)")
ok = True
for task in TASKS:
    tr = G((task,), ("A60", "E_nocopy"), ("T",))
    hm = G((task,), ("A60", "E_nocopy", "E_copy"), ("S",)) + G((task,), ("N60", "R60", "N_same"), ("T", "S"))
    e, lo, hi = cluster_boot(tr, "watch_b40"); ei, loi, hii = item_boot(tr, "watch_b40")
    e2, lo2, hi2 = cluster_boot(hm, "watch_b40"); e2i, lo2i, hi2i = item_boot(hm, "watch_b40")
    g_ok = (e >= 0.10 and lo > 0) if task in ("parcels", "bonus") else None
    h_ok = e2 >= -0.05
    ok &= h_ok and (g_ok is not False)
    print(f"  {task:8} GAIN n={len(tr)}: {e:+.1%} [{lo:+.1%},{hi:+.1%}] (item-boot [{loi:+.1%},{hii:+.1%}]) "
          f"{'' if g_ok is None else ('PASS' if g_ok else 'FAIL')}")
    print(f"  {task:8} HARM n={len(hm)}: {e2:+.1%} [{lo2:+.1%},{hi2:+.1%}] (item-boot [{lo2i:+.1%},{hi2i:+.1%}]) "
          f"{'PASS' if h_ok else 'FAIL'}")
    for pol in ("watch_all", "oracle_b40"):
        a = cluster_boot(tr, pol); b = cluster_boot(hm, pol)
        print(f"           [{pol}] gain {a[0]:+.1%} [{a[1]:+.1%},{a[2]:+.1%}]  harm {b[0]:+.1%} [{b[1]:+.1%},{b[2]:+.1%}]")
print("  OVERALL:", "PASS" if ok else "FAIL")

# per-arm harm breakdown for bonus
print("\n### bonus harm by group (watch_b40 - normal, pp)")
for arm in ("A60", "E_nocopy", "E_copy", "N60", "R60", "N_same"):
    for pr in (("S",) if arm in ("A60", "E_nocopy", "E_copy") else ("T", "S")):
        g = G(("bonus",), (arm,), pr)
        d = np.mean([right(r, "watch_b40") - right(r, "normal") for r in g])
        print(f"  {arm:9} {''.join(pr):3} n={len(g)}  {100*d:+.1f}")

# ---------------------------------------------------------------- watch: leakage diagnostics
print("\n### Leakage diagnostics")
same = [r for r in R if r["flags"] == r["truth"] and any(r["flags"])]
diff = sum(r["scores"]["watch_b40"] != r["scores"]["oracle_b40"] for r in same)
print(f"  rows where flags == truth (non-empty): {len(same)}; watch_b40 scores differ from oracle_b40 in {diff} (must be 0)")
noflag = [r for r in R if not any(r["flags"])]
print(f"  rows with no flag: {len(noflag)}; watch_b40 != normal in {sum(r['scores']['watch_b40'] != r['scores']['normal'] for r in noflag)};"
      f" watch_all != normal in {sum(r['scores']['watch_all'] != r['scores']['normal'] for r in noflag)}")
notruth = [r for r in R if not any(r["truth"])]
print(f"  rows with no true stale: {len(notruth)}; oracle_b40 != normal in {sum(r['scores']['oracle_b40'] != r['scores']['normal'] for r in notruth)}")
# identical-flag groups must give identical watch scores (watch depends only on prompt+flags)
grp = collections.defaultdict(set)
for it, r in zip(items, R):
    grp[(it["prompt"], tuple(r["flags"]))].add(json.dumps(r["scores"]["watch_b40"]))
print(f"  (prompt,flags) groups with >1 distinct watch score: {sum(len(v) > 1 for v in grp.values())}")
# flags consistent across arms for the same (task,set,probe) and same example case? (memo: same recheck text => same flag)
inc = 0; tot = 0
for (task, s, pr), rs in collections.defaultdict(list, {}).items(): pass
byk = collections.defaultdict(list)
for it, r in zip(items, R):
    up = re.search(r"IMPORTANT UPDATE: (.*)\n", it["prompt"]); note = up.group(1) if up else ""
    for (_, c, a), f in zip(EXL.findall(it["prompt"]), r["flags"]):
        byk[(it["task"], it["set"], note, c, a)].append(f)
for v in byk.values():
    tot += 1; inc += len(set(v)) > 1
print(f"  same (set, note, case, recorded answer) -> different flag: {inc} of {tot} (expected 0 by memo; nonzero would mean hidden state)")

# ---------------------------------------------------------------- watch: what the watchman says about the test case itself
print("\n### Test-case copies in history: flag on copies of the test case (watchman re-solves the TEST CASE itself)")
for task in TASKS:
    for arm in ("A60", "E_nocopy", "N60", "R60"):
        for pr in ("T", "S"):
            fl = []; ncp = []
            for it, r in zip(items, R):
                if it["task"] != task or it["arm"] != arm or it["probe"][0] != pr: continue
                test = re.search(r"TEST CASE: (.+)\n", it["prompt"]).group(1)
                cp = [f for (_, c, a), f in zip(EXL.findall(it["prompt"]), r["flags"]) if c == test]
                ncp.append(len(cp)); fl.append(any(cp))
            print(f"  {task:8} {arm:9} {pr}: copies of test case per prompt {np.mean(ncp):5.1f}; copies flagged in {np.mean(fl):4.0%} of prompts")

# near-duplicates: distinct test cases and distinct prompts
print("\n### Diversity of test cases (per task, probe type)")
for task in TASKS:
    for pr in ("T1", "T2", "S1", "S2"):
        cs = {re.search(r"TEST CASE: (.+)\n", it["prompt"]).group(1) for it in items if it["task"] == task and it["probe"] == pr and it["arm"] == "A60"}
        print(f"  {task:8} {pr}: {len(cs):>2} distinct test-case texts over 20 sets", end="")
    print()
rob = [(a, b) for a in items for b in items if a["task"] == "robot" and b["task"] == "robot" and a["set"] == b["set"]
       and a["arm"] == b["arm"] == "A60" and a["probe"] == "T1" and b["probe"] == "T2"]
print("  robot T1 vs T2 same test case in", sum(re.search(r"TEST CASE: (.+)\n", a["prompt"]).group(1) ==
                                            re.search(r"TEST CASE: (.+)\n", b["prompt"]).group(1) for a, b in rob), "of", len(rob), "sets")
pc = collections.Counter(re.search(r"TEST CASE: (.+)\n", it["prompt"]).group(1) for it in items if it["task"] == "parcels" and it["probe"][0] == "T")
print("  parcels trap test cases:", dict(pc))

# ---------------------------------------------------------------- PUBLIC
print("\n" + "=" * 100, "\nWATCH_PUBLIC\n")
LP = [json.loads(l) for l in open(os.path.join(P, "pubwatch_Qwen2.5-14B-Instruct.jsonl"))]
RP = [r for r in LP if "flags" in r]; chp = next(r["checks"] for r in LP if "checks" in r)
print("checks:", chp["n_zero_flag"], chp["max_zero_flag_diff"], "| rows", len(RP), collections.Counter(r["task"] for r in RP))
rp = lambda r, p: float(am(r["scores"][p]) == r["gold"])
for t in ("hindsight", "algebra", "mmlu"):
    g = [r for r in RP if r["task"] == t]
    print(f"\n{t}: n={len(g)} lamp on {np.mean([any(r['flags']) for r in g]):.1%}, examples flagged {np.mean([x for r in g for x in r['flags']]):.1%}")
    for p in ("normal", "watch_b40", "watch_all"):
        print(f"  {p:10} right {100*np.mean([rp(r, p) for r in g]):.2f}%")
    for p in ("watch_b40", "watch_all"):
        d = np.array([rp(r, p) - rp(r, "normal") for r in g]); rng = np.random.default_rng(3)
        xs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
        up_, dn = int((d > 0).sum()), int((d < 0).sum())
        # exact two-sided sign test on discordant pairs
        from math import comb
        nn = up_ + dn; k = min(up_, dn); pv = min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / 2 ** nn) if nn else 1.0
        print(f"  {p} - normal: {100*d.mean():+.2f} [{100*np.percentile(xs,2.5):+.2f}, {100*np.percentile(xs,97.5):+.2f}]"
              f"  fixed {up_} / broken {dn}  sign-test p={pv:.4f}")
    gold = collections.Counter(r["gold"] for r in g)
    pred = {p: collections.Counter(am(r["scores"][p]) for r in g) for p in ("normal", "watch_b40")}
    print(f"  gold class counts {dict(sorted(gold.items()))}; predicted normal {dict(sorted(pred['normal'].items()))}"
          f" watch_b40 {dict(sorted(pred['watch_b40'].items()))}")
    if t == "hindsight":
        # was the gain only from items with flags? and in which direction did answers move?
        mv = collections.Counter((am(r["scores"]["normal"]), am(r["scores"]["watch_b40"]), r["gold"]) for r in g
                                 if am(r["scores"]["normal"]) != am(r["scores"]["watch_b40"]))
        print("  hindsight changed answers (normal->watch, gold):", dict(mv))
        first200 = [r for r in g if r["i"] < 200]; rest = [r for r in g if r["i"] >= 200]
        for nm, gg in (("items 0-199 (seen in public/ run)", first200), ("items 200-314", rest)):
            d = np.mean([rp(r, "watch_b40") - rp(r, "normal") for r in gg])
            print(f"  {nm}: n={len(gg)} delta {100*d:+.1f}")
print("\nfile sha256 (current):")
for f in ("watch/make_watch.py", "watch/modal_watch.py", "watch/score_watch.py", "newtasks/make_new.py", "newtasks2/make_new2.py",
          "newtasks/check_new.py", "newtasks2/check_new2.py", "eras/settings2.json", "watch_public/modal_pub_watch.py",
          "watch_public/score_pub_watch.py", "watch/sets_watch.json"):
    print(" ", sha(os.path.join(V3, f)), f)
