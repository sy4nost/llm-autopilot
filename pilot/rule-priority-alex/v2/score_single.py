"""Score the one-case-per-prompt full run.  python3 -I score_single.py <dir> <results subdir> [<results subdir> ...]
Answer = first label word in the reply. 5 answers per prompt (1 greedy + 4 sampled).
Gate per model and level (arm none): mean over the 6 probes >= 85% and each primary probe >= 85%.
For every level: primary-probe accuracy by arm and paired bootstrap over 30 sets of (arm - none).
Writes scores_single.json.
"""
import json, os, random, re, sys

D = sys.argv[1]
PRIMARY = {"L1": ["Q1", "Q6"], "L2": ["P1", "P3"], "L3": ["P1", "P3"]}
ORDER = ["0.5B", "1.5B", "3B", "7B", "14B"]
ARMS = ["none", "exp", "mixed", "mixed2", "control"]
items = {(x["level"], x["set"], x["arm"], x["probes"][0]["id"]): x
         for x in json.load(open(os.path.join(D, "sets_single.json")))}


def first_label(t, labels):
    w = [x.upper() for x in re.findall(r"[A-Za-z]+", t) if x.upper() in labels]
    return w[0] if w else None


acc = {}  # (model, level, arm, probe, set) -> share right
for sub in sys.argv[2:]:
    for f in sorted(os.listdir(os.path.join(D, sub))):
        res = json.load(open(os.path.join(D, sub, f)))
        m = res["model"].split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
        for a in res["answers"]:
            it = items[(a["level"], a["set"], a["arm"], a["probe_ids"][0])]
            ts = [a["greedy"]] + a["samples"]
            acc[(m, a["level"], a["arm"], a["probe_ids"][0], a["set"])] = \
                sum(first_label(t, it["labels"]) == it["probes"][0]["key"] for t in ts) / len(ts)

rng = random.Random(1)


def mean(v):
    return sum(v) / len(v)


def boot(d):
    b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
    return mean(d), b[100], b[3899]


out = {}
for m in ORDER:
    for lv in ("L1", "L2", "L3"):
        pids = sorted({k[3] for k in acc if k[0] == m and k[1] == lv})
        if not pids:
            continue
        r = {}
        for arm in ARMS:
            if not any(k[:3] == (m, lv, arm) for k in acc):
                continue
            per = {p: mean([acc[(m, lv, arm, p, s)] for s in range(30)]) for p in pids}
            r[arm] = dict(all=mean(list(per.values())), probes=per)
            if arm != "none":
                r[arm]["vs_none"] = {p: boot([acc[(m, lv, arm, p, s)] - acc[(m, lv, "none", p, s)] for s in range(30)])
                                     for p in PRIMARY[lv]}
        r["gate"] = r["none"]["all"] >= .85 and all(r["none"]["probes"][p] >= .85 for p in PRIMARY[lv])
        out.setdefault(m, {})[lv] = r
json.dump(out, open(os.path.join(D, "scores_single.json"), "w"), indent=1)

for m in ORDER:
    if m not in out:
        continue
    for lv, r in out[m].items():
        print(f"\n{m} {lv} {'GATE' if r['gate'] else '(no gate)'}  none: all {r['none']['all']:.0%}, "
              + ", ".join(f"{p} {v:.0%}" for p, v in r["none"]["probes"].items()))
        for arm in ARMS[1:]:
            if arm in r:
                print(f"   {arm.ljust(8)} all {r[arm]['all']:.0%} | " + "; ".join(
                    f"{p} {r[arm]['probes'][p]:.0%} ({d:+.0%} [{lo:+.0%},{hi:+.0%}])"
                    for p, (d, lo, hi) in r[arm]["vs_none"].items()))
