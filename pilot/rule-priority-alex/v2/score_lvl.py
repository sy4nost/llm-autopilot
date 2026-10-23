"""Score v2 level runs.  python3 -I score_lvl.py <dir>   (dir has sets.json and results/)
Gate per model and level: in arm `none`, accuracy over all 6 probes >= 85% AND on each primary probe >= 85%
(all 5 answers per prompt). A model's working level = hardest level that passes the gate.
Effect on that level: primary-probe accuracy per arm, and paired bootstrap (30 sets) of arm - none.
Writes scores_lvl.json.
"""
import json, os, random, re, sys

D = sys.argv[1]
items = {(x["level"], x["set"], x["arm"]): x for x in json.load(open(os.path.join(D, "sets.json")))}
PRIMARY = {"L1": ["Q1", "Q6"], "L2": ["P1", "P3"], "L3": ["P1", "P3"]}
PAT = re.compile(r"P\s*([1-6])\s*[:\-]\s*\**\s*(?:LABEL\s*[:\-]?\s*)?([A-Za-z]+)", re.I)
ORDER = ["0.5B", "1.5B", "3B", "7B", "14B"]
ARMS = ["none", "exp", "mixed", "mixed2", "control"]


def parse(ans, labels):
    got = {}
    for m in PAT.finditer(ans):
        got.setdefault("P" + m.group(1), m.group(2).upper())
    if not got:  # six bare label lines
        lines = [l.strip(" *.").upper() for l in ans.splitlines() if l.strip()]
        if len(lines) >= 6 and all(l in labels for l in lines[:6]):
            got = {f"P{j}": l for j, l in enumerate(lines[:6], 1)}
    return got


def probe_acc(a):
    """per probe id: share of the 5 answers that are right"""
    it = items[(a["level"], a["set"], a["arm"])]
    ts = [a["greedy"]] + a["samples"]
    ps = [parse(t, it["labels"]) for t in ts]
    return {pr["id"]: sum(p.get(pr["pos"]) == pr["key"] for p in ps) / len(ts) for pr in it["probes"]}


def boot(diffs, rng):
    b = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(4000))
    return sum(diffs) / len(diffs), b[100], b[3899]


rng = random.Random(1)
out = {}
for f in sorted(os.listdir(os.path.join(D, "results"))):
    res = json.load(open(os.path.join(D, "results", f)))
    m = res["model"].split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
    acc = {(a["level"], a["set"], a["arm"]): probe_acc(a) for a in res["answers"]}
    out[m] = {}
    for lv in ("L1", "L2", "L3"):
        r = {}
        for arm in ARMS:
            ks = [k for k in acc if k[0] == lv and k[2] == arm]
            if not ks:
                continue
            pids = list(acc[ks[0]].keys())
            per = {p: sum(acc[k][p] for k in ks) / len(ks) for p in pids}
            r[arm] = dict(all=sum(per.values()) / len(per), probes=per,
                          primary=sum(per[p] for p in PRIMARY[lv]) / len(PRIMARY[lv]))
            if arm != "none":
                diffs = {p: [acc[(lv, s, arm)][p] - acc[(lv, s, "none")][p] for s in range(30)] for p in PRIMARY[lv]}
                r[arm]["vs_none"] = {p: boot(d, rng) for p, d in diffs.items()}
        r["gate"] = r["none"]["all"] >= 0.85 and all(r["none"]["probes"][p] >= 0.85 for p in PRIMARY[lv])
        out[m][lv] = r
json.dump(out, open(os.path.join(D, "scores_lvl.json"), "w"), indent=1)

print("== gate (arm none): all-6 accuracy / primary probes; PASS if >=85% ==")
for m in ORDER:
    if m in out:
        print(m.ljust(5), " | ".join(
            f"{lv}: {out[m][lv]['none']['all']:.0%} / " + ",".join(f"{p}={out[m][lv]['none']['probes'][p]:.0%}"
                                                              for p in PRIMARY[lv])
            + (" PASS" if out[m][lv]["gate"] else " fail") for lv in ("L1", "L2", "L3")))
print("\n== per level: primary-probe accuracy by arm; [arm - none, 95% CI] per primary probe ==")
for m in ORDER:
    if m not in out:
        continue
    for lv in ("L1", "L2", "L3"):
        r = out[m][lv]
        cells = []
        for arm in ARMS:
            if arm not in r:
                continue
            s = f"{arm} {r[arm]['primary']:.0%}"
            if "vs_none" in r[arm]:
                s += " (" + "; ".join(f"{p} {d:+.0%} [{lo:+.0%},{hi:+.0%}]" for p, (d, lo, hi) in r[arm]["vs_none"].items()) + ")"
            cells.append(s)
        print(m.ljust(5), lv, ("GATE " if r["gate"] else "     "), " | ".join(cells))
