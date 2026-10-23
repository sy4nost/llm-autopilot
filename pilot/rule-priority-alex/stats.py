"""Paired bootstrap over the 30 sets: difference between arms in P3 (and P1) accuracy, all 5 answers per prompt.
python3 -I stats.py <dir>"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1]
sets = {(x["set"], x["arm"]): x for x in json.load(open(os.path.join(D, "sets.json")))}
PAT = re.compile(r"P\s*([1-6])\s*[:\-]\s*\**\s*(?:LABEL\s*[:\-]?\s*)?([A-Za-z]+)", re.I)
def parse(ans, labels):
    got = {}
    for m in PAT.finditer(ans):
        got.setdefault("P" + m.group(1), m.group(2).upper())
    if not got:
        lines = [l.strip(" *.").upper() for l in ans.splitlines() if l.strip()]
        if len(lines) >= 6 and all(l in labels for l in lines[:6]):
            got = {f"P{j}": l for j, l in enumerate(lines[:6], 1)}
    return got
def acc(a, pid):
    it = sets[(a["set"], a["arm"])]
    pr = [p for p in it["probes"] if p["id"] == pid][0]
    ts = [a["greedy"]] + a["samples"]
    return sum(parse(t, it["labels"]).get(pr["pos"]) == pr["key"] for t in ts) / len(ts)
rng = random.Random(1)
for m in ["3B", "7B", "14B"]:
    d = json.load(open(os.path.join(D, "results", f"Qwen2.5-{m}-Instruct.json")))
    by = {(a["set"], a["arm"]): a for a in d["answers"]}
    for pid in ("P3", "P1"):
        for a1, a2 in (("exp", "none"), ("mixed", "none"), ("exp", "control"), ("exp", "mixed")):
            diffs = [acc(by[(s, a1)], pid) - acc(by[(s, a2)], pid) for s in range(30)]
            boots = sorted(sum(rng.choice(diffs) for _ in range(30)) / 30 for _ in range(4000))
            print(m, pid, f"{a1}-{a2}", f"{sum(diffs)/30:+.0%} [{boots[100]:+.0%}, {boots[3899]:+.0%}]")
    print(m, "set0 (Alex's original) greedy:", {arm: by[(0, arm)]["greedy"].replace("\n", " ") for arm in ("exp", "mixed", "control", "none")})
