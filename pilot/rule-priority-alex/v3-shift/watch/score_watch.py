"""Score the watchman run (PREREG.md). python3 -I score_watch.py watch_Qwen2.5-14B-Instruct.jsonl"""
import json, sys, random
import numpy as np
L = [json.loads(l) for l in open(sys.argv[1])]
chk, R = L[0]["checks"], L[1:]
print("checks: prompts with nothing flagged", chk["n_zero_flag"], "| max diff watch vs normal there", chk["max_zero_flag_diff"], "(must be 0)")
am = lambda v: int(np.argmax(v))
ans = lambda r, pol: r["labels"][am(r["scores"][pol])]
right = lambda r, pol: float(ans(r, pol) == r["key"])
oldans = lambda r, pol: float(ans(r, pol) == r["old"] and r["old"] != r["key"])
def boot(rows, pol, base="normal", B=4000, seed=0):
    sets = sorted({(r["task"], r["set"]) for r in rows}); by = {s: [r for r in rows if (r["task"], r["set"]) == s] for s in sets}
    d = {s: (sum(right(r, pol) for r in by[s]), sum(right(r, base) for r in by[s]), len(by[s])) for s in sets}
    rng = random.Random(seed); xs = []
    for _ in range(B):
        smp = [rng.choice(sets) for _ in sets]; n = sum(d[s][2] for s in smp)
        xs.append((sum(d[s][0] for s in smp) - sum(d[s][1] for s in smp)) / n)
    est = (sum(v[0] for v in d.values()) - sum(v[1] for v in d.values())) / sum(v[2] for v in d.values())
    return est, np.percentile(xs, 2.5), np.percentile(xs, 97.5)
G = lambda task, arms, probes: [r for r in R if r["task"] in task and r["arm"] in arms and r["probe"][0] in probes]
POLS = ["normal", "watch_b40", "watch_all", "oracle_b40"]
TASKS = ("parcels", "bonus", "robot", "points")
for task in TASKS:
    rows = [r for r in R if r["task"] == task]
    f = np.array([x for r in rows for x in r["flags"]]); t = np.array([x for r in rows for x in r["truth"]])
    print(f"\n######## {task}   watchman on single examples: stale caught {f[t].mean():.0%} of {t.sum()}, "
          f"wrongly flagged {f[~t].mean():.1%} of {(~t).sum()}")
    print(f"{'group':26}{'n':>4}{'lamp on':>9}" + "".join(f"{p:>12}" for p in POLS) + f"{'old(norm)':>11}{'old(watch)':>11}")
    for arm in ("A60", "E_nocopy", "E_copy", "N60", "R60", "N_same", "F0"):
        for pr, nm in (("T", "trap"), ("S", "untouched")):
            g = G((task,), (arm,), (pr,))
            if not g: continue
            print(f"{arm+' '+nm:26}{len(g):>4}{np.mean([any(r['flags']) for r in g]):>9.0%}"
                  + "".join(f"{np.mean([right(r, p) for r in g]):>12.0%}" for p in POLS)
                  + f"{np.mean([oldans(r, 'normal') for r in g]):>11.0%}{np.mean([oldans(r, 'watch_b40') for r in g]):>11.0%}")
print("\n######## PRE-REGISTERED VERDICT (watch_b40 vs normal, right answers, paired bootstrap over sets, 95%)")
ok = True
for task in TASKS:
    hm = G((task,), ("A60", "E_nocopy", "E_copy"), ("S",)) + G((task,), ("N60", "R60", "N_same"), ("T", "S"))
    e2, lo2, hi2 = boot(hm, "watch_b40"); h_ok = e2 >= -0.05; ok &= h_ok
    line = f"  {task:8} HARM where experience is right (n={len(hm)}): {e2:+.0%} [{lo2:+.0%}, {hi2:+.0%}] need >= -5% -> {'PASS' if h_ok else 'FAIL'}"
    tr = G((task,), ("A60", "E_nocopy"), ("T",))
    e, lo, hi = boot(tr, "watch_b40")
    if task in ("parcels", "bonus"):
        g_ok = e >= 0.10 and lo > 0; ok &= g_ok
        print(f"  {task:8} GAIN traps A60+E_nocopy (n={len(tr)}): {e:+.0%} [{lo:+.0%}, {hi:+.0%}] need >= +10% and low > 0 -> {'PASS' if g_ok else 'FAIL'}")
    else:
        print(f"  {task:8} (no habit expected) traps A60+E_nocopy: {e:+.0%} [{lo:+.0%}, {hi:+.0%}]")
    print(line)
print("  OVERALL:", "PASS" if ok else "FAIL")
