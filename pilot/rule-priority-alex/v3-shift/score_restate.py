"""Score v3b (restate).  python3 -I score_restate.py <dir>
restate_N:      share of answers whose Rule 2 has the NEW value of the changed field / the OLD value / other
restate_ans_N:  same for the restated rule, plus the label on the LABEL line: right (new rule) / old-rule label;
                and the cross-tab: restated new rule but labelled by the old rule.
Compared with the plain change_N answers from results/ (no restating).
"""
import json, os, re, sys

D = sys.argv[1]
items = {}
for x in json.load(open(os.path.join(D, "sets_restate.json"))):
    items[(x["set"], x["arm"], x["probes"][0]["id"])] = x


def rule_kind(t, meta):
    line = next((l for l in t.splitlines() if "RULE 2" in l.upper()), t.splitlines()[0] if t.strip() else "")
    m = re.search(meta["f4"] + r"\s*(?:=|is)\s*([A-Za-z]+)", line, re.I)
    if not m:
        return "other"
    v = m.group(1).upper()
    return "new" if v == meta["f4_new"] else "old" if v == meta["f4_old"] else "other"


def label(t, labels):
    m = re.search(r"LABEL\s*:\s*\**\s*([A-Za-z]+)", t, re.I)
    if m and m.group(1).upper() in labels:
        return m.group(1).upper()
    return None


res = json.load(open(os.path.join(D, "results_restate", "Qwen2.5-14B-Instruct.json")))
stats = {}
for a in res["answers"]:
    it = items[(a["set"], a["arm"], a["probe_ids"][0])]
    pr = it["probes"][0]
    for t in [a["greedy"]] + a["samples"]:
        st = stats.setdefault(a["arm"], dict(n=0, new=0, old=0, other=0, right=0, oldlab=0, nolab=0,
                                             new_but_old=0, new_n=0, f1f3=dict(n=0, right=0, oldlab=0)))
        st["n"] += 1
        k = rule_kind(t, it["meta"])
        st[k] += 1
        if a["arm"].startswith("restate_ans"):
            lab = label(t, it["labels"])
            st["right"] += lab == pr["key"]
            st["oldlab"] += lab == pr["old"]
            st["nolab"] += lab is None
            if k == "new":
                st["new_n"] += 1
                st["new_but_old"] += lab == pr["old"]
            if pr["id"] in ("D1", "D3"):
                st["f1f3"]["n"] += 1
                st["f1f3"]["right"] += lab == pr["key"]
                st["f1f3"]["oldlab"] += lab == pr["old"]
p = lambda a, b: f"{a / b:.0%}" if b else "-"
print("restated Rule 2 (share of answers): new / old / other")
for n in ("0", "20", "60"):
    for arm in (f"restate_{n}", f"restate_ans_{n}"):
        s = stats[arm]
        print(f"  {arm.ljust(14)} {p(s['new'], s['n'])} / {p(s['old'], s['n'])} / {p(s['other'], s['n'])}")
print("\nrestate then answer, D1-D4: right / old-rule label / no label | D1+D3: right / old-rule label | "
      "restated NEW rule but gave old-rule label")
for n in ("0", "20", "60"):
    s = stats[f"restate_ans_{n}"]
    f = s["f1f3"]
    print(f"  N={n.ljust(3)} {p(s['right'], s['n'])} / {p(s['oldlab'], s['n'])} / {p(s['nolab'], s['n'])} | "
          f"{p(f['right'], f['n'])} / {p(f['oldlab'], f['n'])} | {p(s['new_but_old'], s['new_n'])}")
