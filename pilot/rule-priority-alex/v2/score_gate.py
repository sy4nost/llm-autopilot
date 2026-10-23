"""Score the gate-format run (results_gate/). Arm `none` only. Formats: reason (last P-lines win), single.
python3 -I score_gate.py <dir>
"""
import json, os, re, sys

D = sys.argv[1]
PRIMARY = {"L1": ["Q1", "Q6"], "L2": ["P1", "P3"], "L3": ["P1", "P3"]}
ORDER = ["0.5B", "1.5B", "3B", "7B", "14B"]
items = {}
for x in json.load(open(os.path.join(D, "sets_gate.json"))):
    items[(x["level"], x["set"], x["fmt"], tuple(q["id"] for q in x["probes"]))] = x
PAT = re.compile(r"P\s*([1-6])\s*[:\-]\s*\**\s*(?:LABEL\s*[:\-]?\s*)?([A-Za-z]+)", re.I)


def answers(t, it):
    labels = it["labels"]
    if it["fmt"] == "single":
        w = [x.upper() for x in re.findall(r"[A-Za-z]+", t) if x.upper() in labels]
        return {"P1": w[0] if w else None}
    got = {}
    # each line that starts with P<n>: take the first label word on it; the last such line per P wins
    # (case-echo lines carry no label word; "P3: RULE 1: Label as STOP because ..." gives STOP)
    for line in t.splitlines():
        m = re.match(r"\W*P\s*([1-6])\b(.*)", line.strip())
        if not m:
            continue
        w = [x.upper() for x in re.findall(r"[A-Za-z]+", m.group(2)) if x.upper() in labels]
        if w:
            got["P" + m.group(1)] = w[0]
    return got


acc = {}
for f in sorted(os.listdir(os.path.join(D, "results_gate"))):
    res = json.load(open(os.path.join(D, "results_gate", f)))
    m = res["model"].split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
    for a in res["answers"]:
        it = items[(a["level"], a["set"], a["fmt"], tuple(a["probe_ids"]))]
        for t in [a["greedy"]] + a["samples"]:
            g = answers(t, it)
            for pr in it["probes"]:
                acc.setdefault((m, a["level"], a["fmt"], pr["id"]), []).append(g.get(pr["pos"]) == pr["key"])
print("all-6 accuracy / primary probes (arm none, 5 answers x 30 sets); GATE = all >= 85% and each primary >= 85%")
for fmt in ("reason", "single"):
    print("\n==", fmt, "==")
    for m in ORDER:
        cells = []
        for lv in ("L1", "L2", "L3"):
            ps = {k[3]: sum(v) / len(v) for k, v in acc.items() if k[:3] == (m, lv, fmt)}
            if not ps:
                continue
            allv = sum(ps.values()) / len(ps)
            gate = allv >= .85 and all(ps[p] >= .85 for p in PRIMARY[lv])
            cells.append(f"{lv}: {allv:.0%} / " + ",".join(f"{p}={ps[p]:.0%}" for p in PRIMARY[lv]) + (" GATE" if gate else ""))
        if cells:
            print(m.ljust(5), " | ".join(cells))
