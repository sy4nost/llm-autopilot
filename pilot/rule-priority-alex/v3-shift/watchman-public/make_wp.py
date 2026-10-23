"""Public habit tests never used in this project (fast answers only, for the pilot; later the frozen watchman test).
python3 -I make_wp.py SEED [--pilot] -> sets_wp_pilot.json (pilot sizes) or sets_wp.json
  cpkc   BIG-bench contextual_parametric_knowledge_conflicts: a passage swaps the real answer for another entity;
         "answer assuming the context is true". 2 options: context entity (right) vs the memorized one (habit). No controls.
  modar  BIG-bench modified_arithmetic: 5 worked lines of "a op b -> a op b + 1", then a new line. Options: op+1 (right),
         plain op (habit), op+2 (other). Controls: the *_control subtasks (plain op is right; options op, op+1, op-1).
  base9  Wu et al. 2023 "Reasoning or Reciting" counterfactual arithmetic, their template, fresh 2-digit numbers.
         Traps: base 9, options base-9 sum (right), the base-10 sum written as is (habit), the base-9 sum +-1 (other).
         Controls: the same question in base 10.
Data files in data/ (raw BIG-bench task.json from GitHub); they are read as JSON only."""
import json, os, random, re, sys, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
THINK = "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"
SUBS = ["three_digit_addition", "three_digit_subtraction", "two_digit_multiplication"]


def it(family, pop, prompt, cands, key, habit, cluster):
    assert len(set(cands)) == len(cands)
    return dict(family=family, pop=pop, prompt=prompt, cands=cands, key=key, habit=habit, mode="q", stem="", cluster=cluster)


def shuffled(rng, opts):
    """opts: list of (text, role); returns cands, index of 'right', index of 'habit' (or None)"""
    o = list(opts); rng.shuffle(o)
    c = [t for t, _ in o]
    return c, [r for _, r in o].index("right"), ([r for _, r in o].index("habit") if "habit" in [r for _, r in o] else None)


def cpkc(rng, n, avoid):
    ex = json.load(open(os.path.join(D, "contextual_parametric_knowledge_conflicts.json")))["examples"]
    pre = "What is the answer to the question, assuming the context is true.\n"
    ex = [e for e in ex if len(e["target_scores"]) == 2 and e["input"].strip() not in avoid   # review S1: strip
          and not any(f" {a.lower()} " in f" {b.lower()} " for a in e["target_scores"] for b in e["target_scores"] if a != b)]  # S5: no name inside the other
    out = []
    for e in rng.sample(ex, n):
        right = e["target"]; habit = [k for k in e["target_scores"] if k != right][0]
        c, k, h = shuffled(rng, [(right, "right"), (habit, "habit")])
        q = e["input"].strip()
        out.append(it("cpkc", "trap", pre + q + "\nOptions: " + " / ".join(c), c, k, h, "habit:" + habit))   # S4: cluster = memorized name
    return out


def opkey(p):
    """the question without its answers: row i of *_plus_one and of *_control share operands and shot lines (review S2)"""
    return re.sub(r"-> -?\d+", "->", p)


def modar(rng, n, nc, avoid):
    out = []; av = {opkey(p) for p in avoid}
    for j, sub in enumerate(SUBS):
        P = json.load(open(os.path.join(D, f"ma_{sub}_plus_one.json")))["examples"]
        C = json.load(open(os.path.join(D, f"ma_{sub}_control.json")))["examples"]
        assert len(P) == len(C) and all(opkey(x["input"]) == opkey(y["input"]) for x, y in zip(P, C))
        seen, rows = set(av), []
        for i in rng.sample(range(len(P)), len(P)):   # each operand row used at most once, trap or control
            kq = opkey(P[i]["input"])
            if kq not in seen:
                seen.add(kq); rows.append(i)
        mt = n // 3 + (1 if j < n % 3 else 0); mc = nc // 3 + (1 if j < nc % 3 else 0)
        for pop, ids in (("trap", rows[:mt]), ("control", rows[mt:mt + mc])):
            for i in ids:
                e = (P if pop == "trap" else C)[i]; t = int(e["target"])
                if pop == "trap":
                    c, k, h = shuffled(rng, [(str(t), "right"), (str(t - 1), "habit"), (str(t + 1), "other")])
                else:
                    c, k, h = shuffled(rng, [(str(t), "right"), (str(t + 1), "other"), (str(t - 1), "other2")])
                    h = None
                out.append(it("modar", pop, e["input"], c, k, h, opkey(e["input"])))
    return out


def b9(x, base=9):
    s = ""
    while x:
        s = str(x % base) + s; x //= base
    return s or "0"


def base9(rng, n, nc, avoid=frozenset()):
    out, seen = [], set(avoid)
    T = 'You are a mathematician. Assuming that all numbers are in base-{b}, where the digits are "{d}", what is {x}+{y}?'
    for pop, cnt in (("trap", n), ("control", nc)):
        k = 0
        while k < cnt:
            if pop == "trap":   # digits 0-8, a carry in base 9 so that base-9 and base-10 sums differ
                x = int(f"{rng.randint(1, 8)}{rng.randint(0, 8)}", 9); y = int(f"{rng.randint(1, 8)}{rng.randint(0, 8)}", 9)
                xs, ys, right, habit = b9(x), b9(y), b9(x + y), str(int(b9(x)) + int(b9(y)))
                if right == habit:
                    continue
                other = b9(x + y + rng.choice((-1, 1)))
                if other in (right, habit):
                    continue
                c, ki, h = shuffled(rng, [(right, "right"), (habit, "habit"), (other, "other")])
                p = T.format(b=9, d="012345678", x=xs, y=ys)
            else:
                x, y = rng.randint(10, 88), rng.randint(10, 88)
                right = str(x + y); o1 = str(x + y + 1); o2 = str(x + y - 1)
                c, ki, h = shuffled(rng, [(right, "right"), (o1, "other"), (o2, "other2")]); h = None
                p = T.format(b=10, d="0123456789", x=x, y=y)
            if p in seen:
                continue
            seen.add(p); out.append(it("base9", pop, p, c, ki, h, p)); k += 1
    return out


def build(S, pilot, smoke=False):
    rng = random.Random(S)
    n, nc = (60, 30) if pilot else (4, 2) if smoke else (100, 40)
    avoid = set()
    if not smoke and os.path.exists(os.path.join(HERE, "sets_smoke.json")) and not pilot:
        avoid |= {x["prompt"] for x in json.load(open(os.path.join(HERE, "sets_smoke.json")))["items"]}
    if not pilot and os.path.exists(os.path.join(HERE, "sets_wp_pilot.json")):
        avoid |= {x["prompt"] for x in json.load(open(os.path.join(HERE, "sets_wp_pilot.json")))["items"]}
        avoid |= {p.split("What is the answer to the question, assuming the context is true.\n", 1)[-1].split("\nOptions:")[0] for p in avoid}
    items = cpkc(rng, n, avoid) + modar(rng, n, nc, avoid) + base9(rng, n, nc, avoid)
    seen, out = set(avoid), []
    for x in items:
        if x["prompt"] in seen:
            continue
        seen.add(x["prompt"]); x["id"] = len(out); x["think"] = x["prompt"] + THINK; out.append(x)
    return out


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv; smoke = "--smoke" in sys.argv; S = int([a for a in sys.argv[1:] if not a.startswith("--")][0])
    assert pilot or smoke or S not in (20261024, 12345, 57, 58, 78, 2, 88), "seed already used"
    items = build(S, pilot, smoke)
    fn = os.path.join(HERE, "sets_wp_pilot.json" if pilot else "sets_smoke.json" if smoke else "sets_wp.json")
    json.dump(dict(seed=S, items=items), open(fn, "w"), ensure_ascii=False)
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16])
    print(dict(sorted(Counter((x["family"], x["pop"]) for x in items).items())))
