"""Independent check of sets_new2.json: re-reads every prompt as TEXT (rules, bonus, update / review note, examples, test case),
re-solves every example and the test case from that text alone and compares with the stored key / old answer.
python3 -I check_new2.py sets_new2.json [--mutate]"""
import json, re, sys, collections

def parse_head(p):
    head = p.split("Below are")[0].split("Answer for the TEST")[0]
    if "PARCEL ROUTING" in head:
        lab = re.findall(r"goes by (\w+)\.|parcels go by (\w+)\.", head)
        lab = [a or b for a, b in lab]
        r2 = re.search(r"insured and, at the same time, (express|standard) goes by", head).group(1)
        return "parcels", dict(lab=lab, r2=r2)
    val = {c: int(v) for c, v in re.findall(r"- an? (\w+) token is worth (\d+) point", head)}
    m = re.search(r"BONUS RULE: If a pile contains an? (\w+) token and, at the same time, (no|an?) (\w+) token, add (\d+)", head)
    return "bonus", dict(val=val, x=m.group(1), yneg=m.group(2) == "no", y=m.group(3), b=int(m.group(4)))

def solve(kind, R, case):
    if kind == "parcels":
        heavy = "heavy parcel" in case; abroad = "abroad" in case; ins = "not insured" not in case; exp = "express delivery" in case
        T, A, S, V = R["lab"]
        if heavy: return T
        if ins and (exp if R["r2"] == "express" else not exp): return A
        return S if abroad else V
    cols = re.findall(r"1 (\w+)", case)
    bon = R["x"] in cols and ((R["y"] not in cols) if R["yneg"] else (R["y"] in cols))
    return str(sum(R["val"][c] for c in cols) + R["b"] * bon)

def check(it):
    p, errs = it["prompt"], []
    kind, base = parse_head(p)
    after = dict(base)
    m = re.search(r"IMPORTANT UPDATE: (.*)\n", p)
    review = bool(m) and "reviewed and stays the same" in m.group(1)
    if m:
        t = m.group(1)
        if kind == "parcels":
            r2 = re.search(r"insured and, at the same time, (express|standard) goes by (\w+)", t)
            if r2.group(2) != base["lab"][1]: errs.append("note names another transport")
            if review and r2.group(1) != base["r2"]: errs.append("review note contradicts rules")
            after["r2"] = r2.group(1)
        else:
            b = re.search(r"contains an? (\w+) token and, at the same time, (no|an?) (\w+) token, add (\d+)", t)
            if (b.group(1), b.group(3), int(b.group(4))) != (base["x"], base["y"], base["b"]): errs.append("note names other colours")
            if review and (b.group(2) == "no") != base["yneg"]: errs.append("review note contradicts rules")
            after["yneg"] = b.group(2) == "no"
    upd_pos = m.start() if m else len(p)
    rows = [(mm.start(), mm.group(1), mm.group(2)) for mm in re.finditer(r"Example \d+: (.+?) -> (\w+)", p)]
    for pos, case, ans in rows:
        want = solve(kind, base if pos < upd_pos else after, case)
        if want != ans: errs.append(f"example '{case} -> {ans}' should be {want}")
    test = re.search(r"TEST CASE: (.+)\n", p).group(1)
    first60 = [c for _, c, _ in rows[:60]]
    if it["arm"] != "F0" and first60.count(test) < 2: errs.append(f"test case appears {first60.count(test)}x in first 60")
    it["_copies"] = first60.count(test)
    k = solve(kind, after, test)
    if k != it["key"]: errs.append(f"key {it['key']} but text gives {k}")
    if it["arm"] != "F0":
        if solve(kind, base, test) != it["old"]: errs.append("old mismatch")
        if m and not review and (solve(kind, after, test) != solve(kind, base, test)) != it["changed"]: errs.append("changed flag vs text")
    if not (it["key"] in it["labels"] and it["old"] in it["labels"] and len(set(it["labels"])) == 3): errs.append("labels")
    if it["probe"].startswith("T") != it["changed"]: errs.append("probe type")
    if it["arm"] == "E_nocopy":
        p2 = [c for _, c, _ in rows[60:]]
        if test in p2: errs.append("E_nocopy period 2 contains the test case")
    if it["arm"] == "E_copy" and test not in [c for _, c, _ in rows[60:]]: errs.append("E_copy period 2 lacks the test case")
    return errs

d = json.load(open(sys.argv[1]))["items"]
if "--mutate" in sys.argv:
    d[5]["prompt"] = d[5]["prompt"].replace("-> ", "-> X", 1)
    d[9]["key"] = "ZZ"
    j = next(i for i, x in enumerate(d) if x["task"] == "bonus" and x["arm"] == "A60")
    d[j]["prompt"] = d[j]["prompt"].replace("no ", "a ", 1)
bad = [(i, e) for i, it in enumerate(d) for e in check(it)]
print(len(d), "prompts checked,", len(bad), "problems")
for t in ("parcels", "bonus"):
    c = [it["_copies"] for it in d if it["task"] == t and it["arm"] == "A60"]
    print(f"  {t}: exact copies of the test case among the 60 examples: min {min(c)}, median {sorted(c)[len(c)//2]}, max {max(c)}")
    ch = [it for it in d if it["task"] == t and it["arm"] == "A60" and it["probe"][0] == "T"]
    print(f"  {t}: trap answers old->new e.g.", collections.Counter((x["old"] != x["key"]) for x in ch))
for i, e in bad[:12]: print(" ", i, d[i]["task"], d[i]["arm"], d[i]["probe"], e)
