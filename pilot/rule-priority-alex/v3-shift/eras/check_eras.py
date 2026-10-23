"""Independent data check for sets_eras.json: re-derive every label in every prompt from the rules.
python3 -I check_eras.py"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from make_shift import key, case_str  # noqa: E402
d = json.load(open(os.path.join(HERE, "sets_eras.json")))
m = {(x["set"], x["probes"][0]["id"]): x for x in json.load(open(os.path.join(HERE, "..", "match", "sets_match.json"))) if x["arm"] == "A_change_60"}
bad = 0


def row_ok(prompt, cs, c):
    """Independent of make_shift.case_str: read field names/values from the header line and decode the row."""
    fields = re.search(r"Every case has four fields: (.*?)\. The field", prompt).group(1)
    spec = {n: v.split("/") for n, v in re.findall(r"(\w+) \(([^)]*)\)", fields)}
    r1 = re.search(r"RULE 1 \(highest priority\): Label a case \w+ if its (\w+) is (\w+)", prompt).groups()
    r2 = re.search(r"RULE 2: If Rule 1 does not apply, label the case \w+ if its (\w+) is (\w+) and, at the same time, its (\w+) is (\w+)", prompt).groups()
    r3 = re.search(r"RULE 3: If Rules 1 and 2 do not apply, label the case \w+ if its (\w+) is (\w+)", prompt).groups()
    vals = dict(kv.split("=") for kv in cs.split("; "))
    f2name = r3[0]
    f4name = r2[2]
    # c = (f1 trigger, f2 is the PASS value, f3 present, f4 is the old-rule value)
    want = (vals[r1[0]] == r1[1], vals[f2name] == r3[1], vals[r2[0]] == r2[1], vals[f4name] == r2[3])
    return tuple(bool(v) for v in c) == want
for x in d:
    pr = x["probes"][0]
    if x["arm"] == "A60":
        bad += x["prompt"] != m[(x["set"], pr["id"])]["prompt"]
        continue
    changed = x["arm"] != "N_same"
    lab = x["labels"]
    ex = re.findall(r"Example (\d+): (.*?) -> (\w+)", x["prompt"])
    assert len(ex) == 70, len(ex)
    cases = x["p1"] + x["p2"]
    # rebuild the case->string map with the set's naming by matching rows in order
    s = None
    for (num, cs, l), c in zip(ex, cases):
        new = changed and int(num) > 60
        # row text must be exactly the stored case rendered; check via field=value pairs and the rule from the header
        bad += not row_ok(x["prompt"], cs, c)
        if l != lab["HPR".index(key(c, new))]:
            bad += 1
    test = tuple(x["case"])
    tstr = re.search(r"TEST CASE: (.*)\n", x["prompt"]).group(1)
    in_p2 = [cs for num, cs, _ in ex if int(num) > 60 and cs == tstr]
    in_p1 = [cs for num, cs, _ in ex if int(num) <= 60 and cs == tstr]
    want_copy = x["arm"] in ("E_copy", "N_same")
    bad += (len(in_p2) > 0) != want_copy
    bad += pr["key"] != lab["HPR".index(key(test, changed))]
    bad += ("IMPORTANT UPDATE" in x["prompt"]) != changed
    if pr["id"] in ("D1", "D3"):
        bad += len(in_p1) == 0  # exact old copies must exist in period 1
        if changed:
            bad += pr["key"] == pr["old"]
print("prompts", len(d), "problems", bad)
from collections import Counter
print(Counter((x["arm"], x["probes"][0]["id"], x["probes"][0]["key"] == x["probes"][0]["old"]) for x in d))
