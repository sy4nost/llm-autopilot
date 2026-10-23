"""Score the frozen run (PREREG.md). python3 -I score_new.py new2_Qwen2.5-14B-Instruct.jsonl
Frozen lamp: compare normal with w0 (old examples silenced in the copy heads). Lamp on if the picked answer changes, or its
lead over the runner-up falls below 50% of the normal lead. Frozen policy: lamp on -> up16_all answer, else normal answer."""
import json, sys, random
import numpy as np
L = [json.loads(l) for l in open(sys.argv[1])]
chk, R = L[0]["checks"], L[1:]
print("checks: max w1 diff", chk["max_w1_diff"], "| max up1 diff", chk["max_up1_diff"], "(both must be 0)")
am = lambda v: int(np.argmax(v))
lead = lambda v, p: (lambda a: a[p] - np.delete(a, p).max())(np.asarray(v, float))
def lamp(sc):
    p = am(sc["normal"])
    return am(sc["w0"]) != p or lead(sc["w0"], p) < 0.5 * lead(sc["normal"], p)
def ans(r, pol):
    sc = r["scores"]
    if pol == "gated": return r["labels"][am(sc["up16_all"] if lamp(sc) else sc["normal"])]
    if pol == "gated4": return r["labels"][am(sc["up4_all"] if lamp(sc) else sc["normal"])]
    return r["labels"][am(sc[pol])]
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
POLS = ["normal", "gated", "up16_all", "w0", "gated4"]
for task in ("parcels", "bonus"):
    print(f"\n######## {task}")
    print(f"{'group':34}{'n':>4}{'lamp on':>9}" + "".join(f"{p:>10}" for p in POLS) + f"{'old(norm)':>11}{'old(gated)':>11}")
    for arm in ("A60", "E_nocopy", "E_copy", "N60", "R60", "N_same", "F0"):
        for pr, nm in (("T", "trap"), ("S", "untouched")):
            rows = G((task,), (arm,), (pr,))
            if not rows: continue
            print(f"{arm+' '+nm:34}{len(rows):>4}{np.mean([lamp(r['scores']) for r in rows]):>9.0%}"
                  + "".join(f"{np.mean([right(r, p) for r in rows]):>10.0%}" for p in POLS)
                  + f"{np.mean([oldans(r, 'normal') for r in rows]):>11.0%}{np.mean([oldans(r, 'gated') for r in rows]):>11.0%}")
print("\n######## PRE-REGISTERED VERDICT (gated vs normal, right answers, paired bootstrap over sets, 95%)")
ok_all = True
for task in ("parcels", "bonus"):
    tr = G((task,), ("A60", "E_nocopy"), ("T",))
    hm = G((task,), ("A60", "E_nocopy", "E_copy"), ("S",)) + G((task,), ("N60", "R60", "N_same"), ("T", "S"))
    e, lo, hi = boot(tr, "gated"); g_ok = e >= 0.10 and lo > 0
    e2, lo2, hi2 = boot(hm, "gated"); h_ok = e2 >= -0.05
    ok_all &= g_ok and h_ok
    print(f"  {task:7} GAIN traps A60+E_nocopy (n={len(tr)}): {e:+.0%} [{lo:+.0%}, {hi:+.0%}]  need >= +10% and low > 0 -> {'PASS' if g_ok else 'FAIL'}")
    print(f"  {task:7} HARM where experience is right (n={len(hm)}): {e2:+.0%} [{lo2:+.0%}, {hi2:+.0%}]  need >= -5% -> {'PASS' if h_ok else 'FAIL'}")
    e3, lo3, hi3 = boot(G((task,), ("E_copy",), ("T",)), "gated"); print(f"  {task:7} (secondary) E_copy traps: {e3:+.0%} [{lo3:+.0%}, {hi3:+.0%}]")
print("  OVERALL:", "PASS" if ok_all else "FAIL")
