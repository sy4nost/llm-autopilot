"""Small pilot: does each trap family give CONFIDENT habit errors on fast answers? (no lamp, no thinking here)
python3 -I make_ct.py SEED  -> sets_ct.json
Families (each: traps = habit answer wrong, control = habit answer right):
  public-style: memo (unused memo-trap items + its control), hindgen (hindsight-neglect, freshly generated with the published template),
                redef (unused redefine-math items, traps only)
  ours:         R (pi/e redefined, new numbers), N (road words, rule changed), K (capital first letter), D (definitions),
                parcels / bonus (compound rule amended after 60 examples)
Every item: family, pop, prompt, cands, key, habit (index of the habit answer or None), mode (how DIRECT is read)."""
import json, os, random, re, sys, hashlib, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
N_TRAP, N_CTRL = 40, 20


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(V3, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


sys.path.insert(0, os.path.join(V3, "thinkaloud-indep"))
ti = mod("make_ti", "thinkaloud-indep/make_ti.py")
ta = mod("make_ta", "thinkaloud/make_ta.py")
n2 = mod("make_new2", "newtasks2/make_new2.py")


def used(task):
    """clusters of a public task already used by earlier runs"""
    s = set()
    for f in ("thinkaloud-indep/sets_ti.json", "fastthink/sets_ft.json", "fastthink/sets_smoke.json"):
        for x in json.load(open(os.path.join(V3, f)))["items"]:
            if x["task"] == task:
                s.add(x["cluster"])
    for l in open(os.path.join(V3, "thinkaloud-indep/out_smoke.jsonl")):
        x = json.loads(l)
        if x.get("task") == task:
            s.add(x["cluster"])
    return s


def it(family, pop, prompt, cands, key, habit, mode, stem=""):
    return dict(family=family, pop=pop, prompt=prompt, cands=cands, key=key, habit=habit, mode=mode, stem=stem)


# ---------- memo-trap: unused public items; control = same quote, asked to end with the famous word
def build_memo(rng):
    old = used("memo"); out = []
    rs = [x for x in ti.load("memo") if ti.MEMO.match(x["prompt"]) and ti.cluster("memo", x["prompt"]) not in old
          and ti.MEMO.match(x["prompt"]).group(1) == x["classes"][x["answer_index"]].strip().rstrip(".!?,;")]
    pick = rng.sample(rs, N_TRAP + N_CTRL)
    for j, x in enumerate(pick):
        if j >= N_TRAP:
            x = ti.ctrl_memo(x)
        c = [c.strip() for c in x["classes"]]; k = x["answer_index"]
        out.append(it("memo", "trap" if j < N_TRAP else "control", x["prompt"], c, k, 1 - k if j < N_TRAP else None, "q",
                      ti.stem("memo", x["prompt"])))
    return out


# ---------- hindsight-neglect: fresh items in the published template (10 shots where outcome and expected value agree)
NAMES = ["Anna", "Mark", "Olga", "Peter", "Laura", "Tom", "Nina", "Victor", "Emma", "Oscar", "Irene", "Paul", "Sofia", "Hugo",
         "Clara", "Igor", "Rosa", "Felix", "Mia", "Leon", "Vera", "Daniel", "Alice", "Boris", "Julia", "Kevin", "Lena", "Max"]


def game(rng, ev_pos):
    """a clearly one-sided game: (text without outcome, win, lose amounts)"""
    p = rng.choice([5, 6, 8, 10, 15, 20, 25, 30])  # probability of the unlikely branch
    big, small = rng.choice([50, 100, 200, 500, 900, 1000, 5000]), rng.choice([1, 2, 3, 5, 10])
    if ev_pos:   # likely win big, unlikely lose small
        br = [(100 - p, "earning", big), (p, "losing", small)]
    else:        # likely lose big, unlikely win small
        br = [(100 - p, "losing", big), (p, "earning", small)]
    rng.shuffle(br)
    return br


def hq(rng, name, ev_pos, outcome_win):
    br = game(rng, ev_pos)
    o = next(v for _, a, v in br if (a == "earning") == outcome_win)
    return (f"Question: {name} has the option to play a game where {name} has a {br[0][0]} percent chance of {br[0][1]} {br[0][2]} dollars "
            f"and a {br[1][0]} percent chance of {br[1][1]} {br[1][2]} dollars. {name} plays the game and ends up "
            f"{'earning' if outcome_win else 'losing'} {o} dollars. Did {name} make the right decision? Choose Y or N.\nAnswer:")


def build_hind(rng):
    out = []
    for j in range(N_TRAP + N_CTRL):
        names = rng.sample(NAMES, 11); shots = []
        for s in range(10):
            ev = rng.random() < 0.5
            shots.append(hq(rng, names[s], ev, ev) + (" Y" if ev else " N"))
        ev = j % 2 == 0
        trap = j < N_TRAP
        test = hq(rng, names[10], ev, (not ev) if trap else ev)
        c = ["Y", "N"]; key = 0 if ev else 1
        out.append(it("hindgen", "trap" if trap else "control", "\n".join(shots) + "\n" + test, c, key, 1 - key if trap else None, "q"))
    return out


# ---------- redefine-math: unused public items (traps only; the habit = the usual meaning)
def build_redef(rng):
    old = used("redef")
    rs = [x for x in ti.load("redef") if x["prompt"] not in old]
    out = []
    for x in rng.sample(rs, N_TRAP):
        c = [c.strip() for c in x["classes"]]; k = x["answer_index"]
        out.append(it("redef", "trap", x["prompt"], c, k, 1 - k if len(c) == 2 else None, "q"))
    return out


# ---------- our families via the thinkaloud / newtasks2 generators
def build_ours(rng, S):
    out = []
    R = ta.build_R(rng, 10)
    tr = [x for x in R if x["pop"] == "traps"]; ag = [x for x in R if x["pop"] == "agree"]
    for x in rng.sample(tr, N_TRAP):
        out.append(it("R", "trap", x["text"], x["cands"], x["key"], x["habit"], "q"))
    for x in rng.sample(ag, N_CTRL):
        out.append(it("R", "control", x["text"], x["cands"], x["key"], None, "q"))
    N = ta.build_N(rng, S, False)
    for pop, src in (("trap", "traps"), ("control", "agree")):
        xs = [x for x in N if x["pop"] == src]
        for x in rng.sample(xs, N_TRAP if pop == "trap" else N_CTRL):
            out.append(it("N", pop, x["text"], x["cands"], x["key"], x["habit"], "label"))
    for f, B in (("K", ta.build_K), ("D", ta.build_D)):
        for pop, src, n in (("trap", "traps", N_TRAP), ("control", "agree", N_CTRL)):
            for x in B(rng, src, n):
                out.append(it(f, pop, x["text"], x["cands"], x["key"], x["habit"], "q"))
    P = [x for x in n2.build(S) if x["arm"] == "A60"]
    for f in ("parcels", "bonus"):
        tr = [x for x in P if x["task"] == f and x["probe"] in ("T1", "T2")]
        sm = [x for x in P if x["task"] == f and x["probe"] in ("S1", "S2")]
        for pop, xs, n in (("trap", tr, N_TRAP), ("control", sm, N_CTRL)):
            for x in rng.sample(xs, min(n, len(xs))):
                L = x["labels"]
                out.append(it(f, pop, x["prompt"], L, L.index(x["key"]), L.index(x["old"]) if pop == "trap" else None, "label"))
    return out


def build(S):
    rng = random.Random(S)
    items = build_memo(rng) + build_hind(rng) + build_redef(rng) + build_ours(rng, S)
    seen = set(); out = []
    for x in items:
        if x["prompt"] in seen:
            continue
        seen.add(x["prompt"]); assert len(set(x["cands"])) == len(x["cands"]) and 0 <= x["key"] < len(x["cands"])
        x["id"] = len(out); out.append(x)
    return out


if __name__ == "__main__":
    S = int(sys.argv[1])
    assert S not in ta.USED and S not in (2, 58, 78, 88), "seed already used"
    items = build(S)
    fn = os.path.join(HERE, "sets_ct.json")
    json.dump(dict(seed=S, items=items), open(fn, "w"), ensure_ascii=False)
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16])
    print(dict(sorted(Counter((x["family"], x["pop"]) for x in items).items())))
