"""v3b: does 14B know the changed Rule 2 (restate), and does saying it first fix the answer?
Built from sets_shift.json change_N prompts (N = 0, 20, 60), same sets and histories.
  restate      : instead of classifying, restate the current Rule 2 (no test case)
  restate_ans  : restate the current Rule 2 first, then classify the test case (D1-D4)
python3 -I make_restate.py sets_shift.json sets_restate.json
"""
import json, re, sys

TAIL_RE = re.compile(r"Classify the TEST case below\. Do not explain\. Return only its label\.\n\nTEST CASE: (.*)\n\nReturn only the label\.$")
out = []
for x in json.load(open(sys.argv[1])):
    if not x["arm"].startswith("change_") or not x["probes"][0]["id"].startswith("D"):
        continue
    m = TAIL_RE.search(x["prompt"])
    assert m, x["prompt"][-300:]
    base, case = x["prompt"][:m.start()], m.group(1)
    nm = re.search(r"New RULE 2: .*? if its (\w+) is (\w+) and, at the same time, its (\w+) is (\w+)\.", base)
    old = re.search(r"^RULE 2: .*? its (\w+) is (\w+)\.$", base, re.M)
    meta = dict(f4=nm.group(3), f4_new=nm.group(4), f4_old=old.group(2))
    n = x["arm"].split("_")[1]
    if x["probes"][0]["id"] == "D1":  # one restate prompt per set and N
        out.append(dict(x, arm=f"restate_{n}", max_new=80, meta=meta, probes=[dict(x["probes"][0], id="R")],
                        prompt=base + "Do not classify anything yet. State the CURRENT Rule 2 that applies now, in one "
                        "line, in the form: RULE 2: <label> if <field>=<VALUE> and <field>=<VALUE>."))
    out.append(dict(x, arm=f"restate_ans_{n}", max_new=100, meta=meta,
                    prompt=base + "First state the CURRENT Rule 2 that applies now, in one line, in the form: RULE 2: "
                    "<label> if <field>=<VALUE> and <field>=<VALUE>. Then classify the TEST case below and give its "
                    "label on a second line in the form: LABEL: <label>.\n\nTEST CASE: " + case))
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(out), "prompts")
