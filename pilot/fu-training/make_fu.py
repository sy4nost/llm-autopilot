"""'Fu / good' preference data: teach the model that past examples are hints, not truth.
Every item = prompt + candidate answers; 'key' is right, 'old' is the answer copied from past experience.
Kinds:
  change  - past experience no longer applies (rule/fact/price changed, or the problem itself differs):
            good = key, fu = old (blind copy)
  same    - past experience applies (identical case, nothing changed): good = copied answer, fu = a wrong one
  noisy   - (rules only) some past decisions were wrong; test case is identical to a wrong one: good = by the rules,
            fu = the copied wrong decision
Families:
  rule  (train + test with unseen words)   fact (train + test with unseen names)
  math  (TEST ONLY, never trained)          price (TEST ONLY, never trained)
python3 -I make_fu.py <out_dir>   -> train.json, test.json
"""
import json, random, sys

# ---------- rule family
TRAIN_W = dict(
    f=["crate", "parcel", "drum", "barrel", "sack", "tube", "case", "bundle"],
    v=[("RED", "GREEN"), ("DRY", "WET"), ("HOT", "COLD"), ("FULL", "EMPTY"), ("NEW", "USED"), ("TALL", "SHORT")],
    lab=[("ALPHA", "BETA", "GAMMA"), ("RED", "BLUE", "GOLD"), ("NORTH", "SOUTH", "EAST"), ("ONE", "TWO", "THREE"),
         ("LEFT", "RIGHT", "CENTER"), ("APPLE", "PEAR", "PLUM")])
TEST_W = dict(
    f=["orb", "shard", "token", "rune", "glyph", "sigil", "totem", "relic"],
    v=[("BRIGHT", "DIM"), ("SMOOTH", "ROUGH"), ("HEAVY", "LIGHT"), ("SOFT", "HARD"), ("RAW", "COOKED")],
    lab=[("KAPPA", "SIGMA", "OMEGA"), ("JADE", "ONYX", "OPAL"), ("FIRST", "SECOND", "THIRD"),
         ("MOSS", "FERN", "REED")])


def rule_item(rng, W, kind):
    fn = rng.sample(W["f"], 3)
    vals = rng.sample(W["v"], 3)
    L = rng.choice(W["lab"])
    which = rng.randint(0, 1)  # which rule changes (value flip)

    def lab(c, changed):
        a = [0, 0]
        a[which] = 1 if changed else 0
        if c[0] == a[0]:
            return L[0]
        if c[1] == a[1]:
            return L[1]
        return L[2]
    cstr = lambda c: "; ".join(f"{fn[k]}={vals[k][c[k]]}" for k in range(3))
    allc = [(a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
    diff = [c for c in allc if lab(c, False) != lab(c, True)]
    test = rng.choice(diff) if kind in ("change", "nohist", "fresh") else rng.choice(allc)
    n = 0 if kind in ("nohist", "fresh") else rng.choice([16, 24, 32])
    hist = [rng.choice(allc) for _ in range(n - 2)] + [test, test] if n else []
    rng.shuffle(hist)
    wrong = None
    if kind == "noisy":
        wrong = rng.choice([x for x in L if x != lab(test, False)])
    rows = []
    for j, c in enumerate(hist, 1):
        y = lab(c, False)
        if kind == "noisy" and (c == test or rng.random() < 0.1):
            y = wrong if c == test else rng.choice([x for x in L if x != y])
        rows.append(f"Decision {j:02d}: {cstr(c)} -> {y}")
    r1v = vals[0][1] if (kind == "fresh" and which == 0) else vals[0][0]
    r2v = vals[1][1] if (kind == "fresh" and which == 1) else vals[1][0]
    rules = (f"RULE 1 (highest priority): label {L[0]} if {fn[0]} is {r1v}.\n"
             f"RULE 2: otherwise, label {L[1]} if {fn[1]} is {r2v}.\n"
             f"RULE 3: all other cases are {L[2]}.\n")
    note = ""
    if kind in ("change", "nohist"):
        k = which
        note = (f"NOTICE: Rule {k + 1} has changed. It now applies when {fn[k]} is {vals[k][1]} "
                f"(instead of {vals[k][0]}). Everything else stays the same.\n\n")
    p = ("Classify a case with this fictional system. Each case has the fields "
         + ", ".join(f"{fn[k]} ({'/'.join(vals[k])})" for k in range(3)) + ".\n\n" + rules + "\n"
         + (f"Past decisions made by staff:\n" + "\n".join(rows) + "\n\n" if rows else "") + note
         + f"CASE: {cstr(test)}\nAnswer with the label only.")
    old = lab(test, False) if not rows else [r for r in rows if r.split(": ", 1)[1].rsplit(" -> ", 1)[0] == cstr(test)][0].rsplit(" -> ", 1)[1]
    key = lab(test, kind in ("change", "nohist", "fresh"))
    if kind == "same" or key == old:
        fu = rng.choice([x for x in L if x != key])
    else:
        fu = old
    return dict(fam="rule", kind=kind, prompt=p, key=key, old=old, fu=fu, cands=list(L))


# ---------- fact family
TRAIN_N = ["Mara", "Toby", "Ines", "Karl", "Lena", "Omar", "Rita", "Sven", "Nadia", "Pavel", "Greta", "Hugo"]
TEST_N = ["Yuki", "Bram", "Zoe", "Elif", "Joon", "Wren", "Ayla", "Cato", "Dara", "Fynn"]
THINGS = [("locker", "locker number"), ("desk", "desk number"), ("parking spot", "parking spot number"),
          ("mailbox", "mailbox number")]


def fact_item(rng, names, kind):
    thing, q = rng.choice(THINGS)
    ppl = rng.sample(names, 6)
    nums = rng.sample(range(10, 99), 8)
    val = dict(zip(ppl, nums))
    who = ppl[0]
    recs = []
    for _ in range(rng.choice([12, 18, 24])):
        p = rng.choice(ppl)
        recs.append(f"{p} used {thing} {val[p]}.")
    recs += [f"{who} used {thing} {val[who]}."] * 2
    rng.shuffle(recs)
    rows = [f"Log {j:02d}: {r}" for j, r in enumerate(recs, 1)]
    new = nums[6]
    note = f"NOTICE: from today {who} is assigned {thing} {new}; the old one was given away.\n\n" if kind == "change" else ""
    p = (f"Here is the usage log of the office.\n" + "\n".join(rows) + "\n\n" + note
         + f"Question: which {thing} will {who} use today? Answer with the number only.")
    old = str(val[who])
    key = str(new) if kind == "change" else old
    other = str(val[ppl[1]])
    return dict(fam="fact", kind=kind, prompt=p, key=key, old=old, fu=old if kind == "change" else other,
                cands=[key, other] if kind == "same" else [key, old])


# ---------- math family (test only): the new problem looks like a solved one but differs
MATH = [  # (statement, question, answer, extra clause placed before the question, changed answer)
    ("{n} has {a} boxes with {b} pens in each box.", " How many pens does {n} have?", lambda a, b, c: a * b,
     " {n} gives away {c} pens.", lambda a, b, c: a * b - c),
    ("A bus has {a} rows with {b} seats in each row.", " How many seats can be used?", lambda a, b, c: a * b,
     " {c} seats are broken.", lambda a, b, c: a * b - c),
    ("{n} reads {a} pages a day for {b} days.", " How many pages does {n} read in total?", lambda a, b, c: a * b,
     " On the last day {n} also reads {c} extra pages.", lambda a, b, c: a * b + c),
    ("A farm has {a} pens with {b} sheep in each.", " How many sheep are on the farm now?", lambda a, b, c: a * b,
     " Then {c} more sheep are born.", lambda a, b, c: a * b + c),
    ("{n} buys {a} packs of {b} stickers.", " How many stickers does {n} have now?", lambda a, b, c: a * b,
     " {n} loses {c} of them on the way home.", lambda a, b, c: a * b - c),
]
NAMES_M = ["Anna", "Ben", "Chloe", "Dev", "Ella", "Finn", "Gia", "Hal"]


def math_item(rng, kind):
    idx = rng.sample(range(len(MATH)), 4)
    shots = []
    for t in idx:
        st, q, f, add, g = MATH[t]
        a, b, c, n = rng.randint(3, 9), rng.randint(3, 9), rng.randint(2, 9), rng.choice(NAMES_M)
        shots.append(dict(t=t, a=a, b=b, c=c, n=n, text=(st + q).format(n=n, a=a, b=b), ans=f(a, b, c)))
    s0 = shots[0]
    st, q, f, add, g = MATH[s0["t"]]
    order = shots[:]
    rng.shuffle(order)
    ex = "\n\n".join(f"Problem: {s['text']}\nAnswer: {s['ans']}" for s in order)
    fmt = dict(n=s0["n"], a=s0["a"], b=s0["b"], c=s0["c"])
    if kind == "change":
        test = (st + add + q).format(**fmt)
        key = g(s0["a"], s0["b"], s0["c"])
    elif kind == "subtle":
        a2 = rng.choice([x for x in range(3, 10) if x != s0["a"]])
        test = (st + q).format(n=s0["n"], a=a2, b=s0["b"])
        key = f(a2, s0["b"], 0)
    else:
        test = s0["text"]
        key = s0["ans"]
    p = ("Here are some problems that were solved earlier.\n\n" + ex + "\n\nNow solve this problem. "
         "Answer with the number only.\nProblem: " + test)
    old, other = s0["ans"], s0["a"] + s0["b"]
    return dict(fam="math", kind=kind, prompt=p, key=str(key), old=str(old),
                cands=[str(key), str(old)] if kind != "same" else [str(key), str(other)])


# ---------- price family (test only): a tariff changed
GOODS = [("shipping", "kg", "parcel of {q} kg"), ("parking", "hour", "stay of {q} hours"),
         ("printing", "page", "job of {q} pages"), ("rental", "day", "rental of {q} days")]


def price_item(rng, kind):
    srv, unit, what = rng.choice(GOODS)
    rate = rng.randint(3, 9)
    new = rng.choice([r for r in range(3, 13) if r != rate])
    qs = [rng.randint(2, 9) for _ in range(6)]
    tq = qs[0]
    qs.append(tq)
    rng.shuffle(qs)
    ex = "\n".join(f"Order {j}: {what.format(q=q)} -> ${q * rate}" for j, q in enumerate(qs, 1))
    note = f"\nNOTICE: the {srv} price is now ${new} per {unit}.\n" if kind == "change" else ""
    top = f"NOTICE: the {srv} price is now ${new} per {unit}.\n\n" if kind == "top" else ""
    p = (top + f"Past {srv} orders and what was charged:\n{ex}\n{note}\n"
         f"New order: {what.format(q=tq)}. How many dollars is charged? Answer with the number only.")
    old = tq * rate
    key = tq * new if kind in ("change", "top") else old
    other = tq * rate + rate
    return dict(fam="price", kind=kind, prompt=p, key=str(key), old=str(old),
                cands=[str(key), str(old)] if kind != "same" else [str(key), str(other)])


def build():
    rng = random.Random(20261017)
    train, test = [], []
    for _ in range(600):
        train.append(rule_item(rng, TRAIN_W, "change"))
    for _ in range(350):
        train.append(rule_item(rng, TRAIN_W, "same"))
    for _ in range(250):
        train.append(rule_item(rng, TRAIN_W, "noisy"))
    for _ in range(250):
        train.append(fact_item(rng, TRAIN_N, "change"))
    for _ in range(150):
        train.append(fact_item(rng, TRAIN_N, "same"))
    rng.shuffle(train)
    for kind, n in (("change", 80), ("same", 40), ("noisy", 40)):
        test += [rule_item(rng, TEST_W, kind) for _ in range(n)]
    for kind, n in (("change", 40), ("same", 20)):
        test += [fact_item(rng, TEST_N, kind) for _ in range(n)]
    for kind, n in (("change", 80), ("same", 40)):
        test += [math_item(rng, kind) for _ in range(n)]
        test += [price_item(rng, kind) for _ in range(n)]
    test += [rule_item(rng, TEST_W, "nohist") for _ in range(40)]
    test += [rule_item(rng, TEST_W, "fresh") for _ in range(40)]
    test += [math_item(rng, "subtle") for _ in range(80)]
    test += [price_item(rng, "top") for _ in range(80)]
    return train, test


if __name__ == "__main__":
    tr, te = build()
    json.dump(tr, open(sys.argv[1] + "/train.json", "w"), indent=1)
    json.dump(te, open(sys.argv[1] + "/test.json", "w"), indent=1)
    from collections import Counter
    print("train", len(tr), Counter((x["fam"], x["kind"]) for x in tr))
    print("test", len(te), Counter((x["fam"], x["kind"]) for x in te))
    print("max chars", max(len(x["prompt"]) for x in tr + te))
