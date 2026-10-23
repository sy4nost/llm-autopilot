"""Reviewer check of redefine-math templates vs the PLAN's disguise / R-same rules.
python3 -I check_redefine.py <redefine-math.jsonl>   (data treated as untrusted: parsed only, never executed)"""
import json, re, sys, random
from collections import Counter, defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
SPLIT_SEED = 12345
dev = set()
for t in range(len(rows) // 100):
    dev |= set(random.Random(SPLIT_SEED + t).sample(range(t * 100, t * 100 + 100), 10))

def tmpl(p):
    return re.sub(r"\d+", "#", p)

def disguise(p):  # PLAN rules applied literally and globally
    q = p.replace("π", "zorp")
    q = re.sub(r"\be\b", "quib", q)
    q = q.replace("+", "⊕").replace("÷", "⊘")
    return q

def safe_eval(expr):
    m = re.fullmatch(r"(\d+)([+÷])(\d+)", expr)
    if m:
        a, op, b = int(m[1]), m[2], int(m[3])
        return a + b if op == "+" else a / b
    m = re.fullmatch(r"(\d+) mod (\d+)", expr)
    if m:
        return int(m[1]) % int(m[2])
    m = re.fullmatch(r"(\d+)", expr)
    if m:
        return int(m[1])
    raise ValueError(expr)

def first_digit(x):
    s = str(x).lstrip("-")
    s = s.replace(".", "").lstrip("0") or "0"
    return s[0]

out = defaultdict(Counter)
examples = {}
for i, r in enumerate(rows):
    t = i // 100
    p, cl, g = r["prompt"], [c.strip() for c in r["classes"]], r["answer_index"]
    out[t]["n"] += 1
    out[t]["tmpl:" + tmpl(p)] += 1
    if cl[0] == cl[1]:
        out[t]["identical_classes"] += 1
    pre, q = p.split("Q:", 1)
    m = re.search(r"first digit of (.+?)\? A:", q)
    expr = m[1]
    # what is gold under the redefinition, and normal-math answer for R-same
    if expr in ("π", "e"):
        d = re.search(r"Redefine [πe] as (.+?)\. Q:", p)[1]
        val = safe_eval(d)
        red = first_digit(val)
        normal = "3" if expr == "π" else "2"
        if "+" in d:
            out[t]["plus_in_definition(disguise breaks it)"] += 1
        if isinstance(val, int) and val == 0:
            out[t]["redef_value_zero"] += 1
        if red == normal:
            out[t]["redef_first_digit==habit"] += 1
    else:
        red = expr[0]  # first character of the expression as text
        val = safe_eval(expr)
        normal = first_digit(val)
        if isinstance(val, float) and not val.is_integer():
            out[t]["division_non_integer"] += 1
    if red == normal:
        out[t]["text_answer==math_answer"] += 1
    if cl[g] != red:
        out[t]["gold!=reviewer_redefined_answer"] += 1
    if cl[1 - g] != normal:
        out[t]["nongold!=normal_math(R-same key wrong)"] += 1
    rs = "Q:" + q
    if "Q:" + q == p:
        out[t]["nothing_before_Q"] += 1
    dz = disguise(p)
    if dz.count("⊕") and t not in (0, 1, 2):
        out[t]["plus_replaced_outside_plus_templates"] += 1
    if t == 0 and "⊕" in dz:
        out[t]["T0_question_changed_by_plus_rule"] += 1
    if re.search(r"\be\b", p) and t != 6:
        out[t]["e_hit_outside_T6"] += 1
    if i in dev:
        out[t]["dev"] += 1
    examples.setdefault(t, (p, dz, "Q:" + disguise(q), cl, g))

for t in sorted(out):
    print(f"=== template {t}")
    for k, v in out[t].items():
        print(f"   {k}: {v}")
    p, dz, rsd, cl, g = examples[t]
    print("   orig    :", p)
    print("   disguise:", dz)
    print("   R-same disguised:", rsd)
