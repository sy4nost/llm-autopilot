"""Independent check of sets_new.json: re-reads every prompt as TEXT (dictionary / values, update note, examples, test case),
re-solves every example and the test case from that text alone, and compares with the stored key / old answer.
Also checks: test case appears at least twice (robot: naturally many times) in the first 60 examples; labels contain key and old; 3 distinct labels.
python3 -I check_new.py sets_new.json [--mutate]"""
import json, re, sys

def parse_rules(p):
    if "ROBOT" in p[:60]:
        return "robot", dict(re.findall(r"- the word '(\w+)' means (\w+)", p.split("Answer for the TEST")[0].split("Below are")[0]))
    return "points", {c: int(v) for v, c in [(v, c) for c, v in re.findall(r"- a (\w+) token is worth (\d+) point", p.split("Below are")[0].split("Answer for the TEST")[0])]}

def solve(kind, rules, case):
    if kind == "robot":
        return rules[re.match(r"command '(\w+)'", case).group(1)]
    return str(sum(int(k) * rules[c] for k, c in re.findall(r"(\d+) (\w+)", case)))

def check(it):
    p, errs = it["prompt"], []
    kind, base = parse_rules(p)
    cur = dict(base)
    m = re.search(r"IMPORTANT UPDATE: (.*)\n", p)
    after = dict(base)
    if m:
        t = m.group(1)
        if "reviewed and stays the same" in t:
            r = re.search(r"'(\w+)' has been reviewed and stays the same: it means (\w+)", t) or re.search(r"a (\w+) token has been reviewed and stays the same: (\d+) point", t)
            if r and str(base[r.group(1)]) != r.group(2): errs.append("review note contradicts rules")
        else:
            r = re.search(r"the word '(\w+)' means (\w+)", t) or re.search(r"a (\w+) token is worth (\d+) point", t)
            after[r.group(1)] = r.group(2) if kind == "robot" else int(r.group(2))
    upd_pos = m.start() if m else len(p)
    rows = [(mm.start(), mm.group(1), mm.group(2)) for mm in re.finditer(r"Example \d+: (.+?) -> (\w+)", p)]
    for pos, case, ans in rows:
        want = solve(kind, base if pos < upd_pos else after, case)
        if want != ans: errs.append(f"example '{case} -> {ans}' should be {want}")
    test = re.search(r"TEST CASE: (.+)\n", p).group(1)
    first60 = [c for _, c, _ in rows[:60]]
    if it["arm"] != "F0" and first60.count(test) < 2: errs.append(f"test case appears {first60.count(test)}x in first 60")
    it["_copies"] = first60.count(test)
    key = solve(kind, after if (m and "reviewed" not in m.group(1)) else base, test)
    if key != it["key"]: errs.append(f"key {it['key']} but text gives {key}")
    if solve(kind, base if it["arm"] != "F0" else base, test) != it["old"] and it["arm"] != "F0": errs.append("old mismatch")
    if not (it["key"] in it["labels"] and it["old"] in it["labels"] and len(set(it["labels"])) == 3): errs.append("labels")
    if it["probe"].startswith("T") != it["changed"]: errs.append("probe type")
    if m and "reviewed" not in m.group(1) and (solve(kind, after, test) != solve(kind, base, test)) != it["changed"]: errs.append("changed flag vs text")
    return errs

d = json.load(open(sys.argv[1]))["items"]
if "--mutate" in sys.argv:   # must be caught: flip one example answer and one key
    d[5]["prompt"] = d[5]["prompt"].replace("-> ", "-> X", 1)
    d[9]["key"] = "ZZ"
bad = [(i, e) for i, it in enumerate(d) for e in check(it)]
print(len(d), "prompts checked,", len(bad), "problems")
import collections
for t in ("robot", "points"):
    c = [it.get("_copies", 0) for it in d if it["task"] == t and it["arm"] == "A60"]
    print(f"  {t}: exact copies of the test case among the 60 examples: min {min(c)}, median {sorted(c)[len(c)//2]}, max {max(c)}")
for i, e in bad[:10]: print(" ", i, d[i]["task"], d[i]["arm"], d[i]["probe"], e)
