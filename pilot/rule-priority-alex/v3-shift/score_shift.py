"""Score the v3 shift run.  python3 -I score_shift.py <dir>
Per arm and probe: share right (key), share answering the OLD-rule label where old != new (habit answers).
Paired bootstrap over 30 sets for change_60 - change_0 and change_20 - change_0 on the four D probes.
"""
import json, os, random, re, sys

D = sys.argv[1]
items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(os.path.join(D, "sets_shift.json")))}
ARMS = ["fresh", "change_0", "change_20", "change_60", "nochange_0", "nochange_20", "nochange_60"]
PIDS = ["D1", "D2", "D3", "D4", "A1", "A2"]
DIS = ["D1", "D2", "D3", "D4"]


def first_label(t, labels):
    w = [x.upper() for x in re.findall(r"[A-Za-z]+", t) if x.upper() in labels]
    return w[0] if w else None


rng = random.Random(1)
for f in sorted(os.listdir(os.path.join(D, "results"))):
    res = json.load(open(os.path.join(D, "results", f)))
    right, old = {}, {}
    for a in res["answers"]:
        it = items[(a["set"], a["arm"], a["probe_ids"][0])]
        pr = it["probes"][0]
        ans = [first_label(t, it["labels"]) for t in [a["greedy"]] + a["samples"]]
        right[(a["arm"], pr["id"], a["set"])] = sum(x == pr["key"] for x in ans) / len(ans)
        old[(a["arm"], pr["id"], a["set"])] = sum(x == pr["old"] for x in ans) / len(ans)
    mean = lambda v: sum(v) / len(v)
    print(res["model"])
    print("share right per probe (D = old and new rule disagree; A = agree)")
    print("arm".ljust(12), " ".join(p.rjust(5) for p in PIDS), " | D mean | old-rule answers on D (change arms)")
    for arm in ARMS:
        r = {p: mean([right[(arm, p, s)] for s in range(30)]) for p in PIDS}
        line = arm.ljust(12) + " " + " ".join(f"{r[p]:5.0%}" for p in PIDS) + f" | {mean([r[p] for p in DIS]):5.0%}"
        if arm.startswith("change"):
            line += f" | {mean([old[(arm, p, s)] for p in DIS for s in range(30)]):.0%}"
        print(line)
    for a1 in ("change_20", "change_60"):
        d = [mean([right[(a1, p, s)] for p in DIS]) - mean([right[("change_0", p, s)] for p in DIS]) for s in range(30)]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"{a1} - change_0 on D probes: {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
