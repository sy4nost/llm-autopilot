"""Fresh items for the think-aloud test (PLAN.md v2).  python3 -I make_ta.py S [--small]  -> sets_ta.json (or sets_smoke.json)
Every item: family, pop, cell, grp, cands, key, habit (index or None), and prompts per arm:
direct, instr, think, thinki (+ disg for R, truth for items whose first sentence is a counterfactual)."""
import json, os, re, sys, random, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
USED = {12345, 12346, 25, 32, 185, 20261018, 20261020, 0, 90} | set(range(88, 99))
INSTR = "Follow the facts stated in this task exactly, even if they contradict what you know."
THINK = "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"
TRUTH = "\n\nIs the first sentence of this task true in the real world? Reply yes or no."
_spec = importlib.util.spec_from_file_location("make_dis", os.path.join(V3, "disguise", "make_dis.py"))
md = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(md)

fd = lambda x: str(x)[0]


# ---------------- R: redefine-math templates (wording copied from the dataset), new numbers
def r_item(rng, t):
    """returns (prompt, cands, key, habit)"""
    while True:
        if t == 3:
            a, b = rng.randint(10, 999), rng.randint(2, 400); g = fd(a + b)
            p = f"Redefine π as {a}+{b}. Q: What is the first digit of π? A:"; h = "3"
        elif t == 5:
            a, b = rng.randint(100, 999), rng.randint(11, 60); v = a % b; g = fd(v) if v else None
            p = f"Redefine π as {a} mod {b}. Q: What is the first digit of π? A:"; h = "3"
        elif t == 6:
            a, b = rng.randint(1, 60), rng.randint(1, 99); g = fd(a + b)
            p = f"Redefine e as {a}+{b}. Q: What is the first digit of e? A:"; h = "2"
        elif t in (7, 8):
            b = rng.randint(2, 9); q = rng.randint(2, 99 // b); a = b * q; g, h = fd(a), fd(q)
            mid = "View ÷ as a random digit." if t == 7 else "View ÷ as a random digit instead of a mathematical operation."
            p = f"{mid} Q: What is the first digit of {a}÷{b}? A:"
        elif t == 4:
            n = rng.randint(100, 999); g, h = fd(n), "3"
            p = f"Redefine π as {n}. Q: What is the first digit of π? A:"
        elif t in (0, 1, 2):
            a, b = rng.randint(1, 60), rng.randint(1, 99); g, h = fd(a), fd(a + b)
            mid = {0: "View number as text. Do not perform computation.", 1: "View + as a random digit instead of a mathematical operation.",
                   2: f"View + as the digit {rng.randint(1, 9)} instead of a mathematical operation."}[t]
            p = f"{mid} Q: What is the first digit of {a}+{b}? A:"
        if g is not None and g != h:
            c = [g, h]; rng.shuffle(c)
            return p, c, c.index(g), c.index(h)


def build_R(rng, k):
    out = []
    for t, n in ((3, 4 * k), (5, 4 * k), (6, 4 * k), (7, 4 * k)):
        for _ in range(n):
            p, c, key, hab = r_item(rng, t); out.append(dict(family="R", pop="traps", cell=f"T{t}", text=p, cands=c, key=key, habit=hab))
    for t in (0, 1, 2, 4, 8):
        for _ in range(int(5 * k / 3)):
            p, c, key, hab = r_item(rng, t); out.append(dict(family="R", pop="cfright", cell=f"T{t}", text=p, cands=c, key=key, habit=hab))
    for _ in range(2 * k):   # R-agree: gold = the habit digit
        n = rng.randint(300, 399); o = str(rng.choice([d for d in range(1, 10) if d != 3])); c = ["3", o]; rng.shuffle(c)
        out.append(dict(family="R", pop="agree", cell="T4a", text=f"Redefine π as {n}. Q: What is the first digit of π? A:", cands=c, key=c.index("3"), habit=None))
        n = rng.randint(200, 299); o = str(rng.choice([d for d in range(1, 10) if d != 2])); c = ["2", o]; rng.shuffle(c)
        out.append(dict(family="R", pop="agree", cell="T6a", text=f"Redefine e as {n}. Q: What is the first digit of e? A:", cands=c, key=c.index("2"), habit=None))
    for x in [x for x in out if x["pop"] == "traps" and x["cell"] in ("T3", "T5", "T6")]:   # paired compute twin, same numbers
        expr = re.search(r"as (.+?)\. Q:", x["text"]).group(1)
        x["twin"] = len(out)
        out.append(dict(family="R", pop="compute", cell="C" + x["cell"][1:], text=f"Q: What is the first digit of {expr}? A:",
                        cands=x["cands"], key=x["key"], habit=None, twin_of=out.index(x)))
    for x in out:
        x["disg"] = md.disguise_R(x["text"])
    return out


# ---------------- N: road words (native change_0), 2 fresh generator seeds
def build_N(rng, S, small):
    out = []
    for g, sd in enumerate(md.seeds(S, 2) if not small else [12345]):
        assert small or sd not in USED
        for a in md.get("native", sd):
            if a["arm"] != "change_0" or (small and a["set"] >= 2):
                continue
            base = dict(family="N", grp=f"{g}-{a['set']}", cands=a["labels"], cell=a["probe"])
            pop = "traps" if a["probe"] in ("D1", "D3") else "agree"
            out.append(dict(base, pop=pop, text=a["prompt"], key=a["labels"].index(a["key"]), habit=a["labels"].index(a["old"]) if pop == "traps" else None))
            out.append(dict(base, pop="confirm", text=md.CHANGED.sub(md.CONFIRM, a["prompt"]), key=a["labels"].index(a["old"]), habit=None))
    conf = [x for x in out if x["pop"] == "confirm"]
    keep = set(id(x) for x in rng.sample(conf, min(len(conf), 60 if not small else 4)))
    return [x for x in out if x["pop"] != "confirm" or id(x) in keep]


# ---------------- K / D (asktwice K v1 / D v1, widened)
LEADS = ["In this world,", "In this story,", "In this game,", "In this puzzle,", "In this tale,", "In this quiz,", "In this book,",
         "In this version of the world,"]
K = [("France", "Paris", "Lyon", "Marseille"), ("Germany", "Berlin", "Munich", "Hamburg"), ("Italy", "Rome", "Milan", "Naples"),
     ("Spain", "Madrid", "Barcelona", "Seville"), ("Japan", "Tokyo", "Osaka", "Kyoto"), ("Russia", "Moscow", "Kazan", "Novosibirsk"),
     ("Egypt", "Cairo", "Alexandria", "Luxor"), ("Brazil", "Brasilia", "Recife", "Salvador"), ("Canada", "Ottawa", "Toronto", "Montreal"),
     ("Australia", "Canberra", "Sydney", "Melbourne"), ("China", "Beijing", "Shanghai", "Wuhan"), ("Turkey", "Ankara", "Istanbul", "Izmir"),
     ("Poland", "Warsaw", "Krakow", "Gdansk"), ("Greece", "Athens", "Thessaloniki", "Patras"), ("Sweden", "Stockholm", "Gothenburg", "Malmo"),
     ("Portugal", "Lisbon", "Porto", "Braga"), ("Kenya", "Nairobi", "Mombasa", "Kisumu"), ("Norway", "Oslo", "Bergen", "Trondheim"),
     ("Austria", "Vienna", "Salzburg", "Graz"), ("Argentina", "Buenos Aires", "Cordoba", "Rosario"),
     ("Netherlands", "Amsterdam", "Rotterdam", "Utrecht"), ("Belgium", "Brussels", "Antwerp", "Ghent"), ("Switzerland", "Bern", "Zurich", "Geneva"),
     ("Ireland", "Dublin", "Cork", "Galway"), ("Finland", "Helsinki", "Tampere", "Turku"), ("Denmark", "Copenhagen", "Aarhus", "Odense"),
     ("Hungary", "Budapest", "Debrecen", "Szeged"), ("Czechia", "Prague", "Brno", "Ostrava"), ("Romania", "Bucharest", "Cluj", "Iasi"),
     ("Ukraine", "Kyiv", "Lviv", "Odesa"), ("India", "New Delhi", "Mumbai", "Kolkata"), ("Pakistan", "Islamabad", "Karachi", "Lahore"),
     ("Indonesia", "Jakarta", "Surabaya", "Bandung"), ("Thailand", "Bangkok", "Phuket", "Pattaya"), ("Vietnam", "Hanoi", "Danang", "Saigon"),
     ("Peru", "Lima", "Cusco", "Arequipa"), ("Chile", "Santiago", "Valparaiso", "Concepcion"), ("Colombia", "Bogota", "Medellin", "Cali"),
     ("Morocco", "Rabat", "Casablanca", "Marrakesh"), ("Nigeria", "Abuja", "Lagos", "Kano"), ("Iran", "Tehran", "Isfahan", "Shiraz")]
KQ = ["What is the first letter of the capital of {c}?", "The capital of {c} starts with which letter?",
      "Which letter does the capital of {c} begin with?", "Name the first letter of the capital of {c}.",
      "What letter does the name of the capital of {c} start with?"]
D = [("triangle", "sides", 3), ("square", "sides", 4), ("pentagon", "sides", 5), ("hexagon", "sides", 6), ("octagon", "sides", 8),
     ("rectangle", "sides", 4), ("cube", "faces", 6), ("spider", "legs", 8), ("ant", "legs", 6), ("beetle", "legs", 6), ("dog", "legs", 4),
     ("cat", "legs", 4), ("horse", "legs", 4), ("bird", "legs", 2), ("chicken", "legs", 2), ("octopus", "arms", 8), ("starfish", "arms", 5),
     ("bicycle", "wheels", 2), ("tricycle", "wheels", 3), ("car", "wheels", 4), ("motorcycle", "wheels", 2), ("hand", "fingers", 5)]


def bal(rng, L, n):
    out = []
    while len(out) < n:
        r = list(L); rng.shuffle(r); out += r
    return out[:n]


def three(rng, gold, habit, dist):
    c = [gold, habit, dist]; assert len(set(c)) == 3, c
    rng.shuffle(c); return c


K_USE = K[:20]     # the 20 dev2 countries (allocation set with dev2 rates, PLAN v2.1)
D_USE = [x for x in D if x[0] in ("triangle", "square", "hexagon", "cube", "spider", "ant", "dog", "octopus", "bicycle",
                                  "tricycle", "hand", "pentagon", "octagon", "rectangle", "starfish")]


def build_K(rng, pop, n):
    out = []
    for c, cap, o1, o2 in bal(rng, K_USE, n):
        fakes = [x for x in (o1, o2) if x[0] != cap[0]]
        fake = rng.choice(fakes); third = o2 if fake == o1 else o1
        val = cap if pop == "agree" else fake
        ctx = f"{rng.choice(LEADS)} the capital of {c} is {val}. " if pop != "plain" else ""
        q = rng.choice(KQ).format(c=c)
        L = sorted({chr(i) for i in range(65, 91)} - {fake[0], cap[0], third[0]})
        if pop == "traps":
            d = third[0] if third[0] not in (fake[0], cap[0]) else rng.choice(L)
            cands = three(rng, fake[0], cap[0], d); key, hab = cands.index(fake[0]), cands.index(cap[0])
        else:
            d1, d2 = rng.sample(L, 2); cands = three(rng, cap[0], d1, d2); key, hab = cands.index(cap[0]), None
        out.append(dict(family="K", pop=pop, cell=c, text=f"{ctx}Q: {q} A:", cands=cands, key=key, habit=hab, cf=pop == "traps"))
    return out


def fake_val(rng, real):
    while True:
        v = rng.choice([real - 2, real - 1, real + 1, real + 2, real + 3, real * 2, max(2, real // 2)])
        if v > 1 and v != real:
            return v


def build_D(rng, pop, n):
    out = []
    for hk, part, real in bal(rng, D_USE, n):
        art = "an" if hk[0] in "aeiou" else "a"
        v = fake_val(rng, real) if pop == "traps" else real
        ctx = f"{rng.choice(LEADS)} {art} '{hk}' has {v} {part}. " if pop != "plain" else ""
        q = rng.choice([f"How many {part} does {art} {hk} have?", f"{art.capitalize()} {hk} has how many {part}?",
                        f"Count the {part} of {art} {hk}. How many are there?", f"What is the number of {part} of {art} {hk}?"])
        if pop == "traps":
            d = next(x for x in (v + real + 1, v + real + 2, 2 * v + 1) if x not in (v, real))
            cands = three(rng, str(v), str(real), str(d)); key, hab = cands.index(str(v)), cands.index(str(real))
        else:
            d1 = next(x for x in (real + 1, real + 2) if x != real); d2 = next(x for x in (real * 2 + 1, real + 3) if x not in (real, d1))
            cands = three(rng, str(real), str(d1), str(d2)); key, hab = cands.index(str(real)), None
        out.append(dict(family="D", pop=pop, cell=hk, text=f"{ctx}Q: {q} A:", cands=cands, key=key, habit=hab, cf=pop == "traps"))
    return out


def old_texts():
    s = set()
    for f in ("../asktwice/sets_dev.json", "../asktwice/sets_dev2.json", "sets_smoke.json"):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            s |= {x["text"].replace("  ", " ") for x in json.load(open(fp))["items"]}
    s |= {json.loads(l)["prompt"].replace("\r\n", "\n") for l in open(os.path.join(V3, "indep", "data", "redefine-math.jsonl"))}
    return s


def dedupe(items, old=frozenset()):
    seen, out = set(old), []
    for x in items:
        if x["text"] not in seen:
            seen.add(x["text"]); out.append(x)
    return out


def prompts(x):
    t = x["text"]
    if x["family"] == "N":
        x["direct"] = t
        x["instr"] = t.replace("Classify the TEST case below.", INSTR + "\n\nClassify the TEST case below.", 1)
        th = t.replace("Classify the TEST case below. Do not explain. Return only its label.", "Classify the TEST case below.")
        th = th.replace("\n\nReturn only the label.", "")
        assert "Return only" not in th and "Do not explain" not in th, th[-200:]
        x["think"] = th + THINK
        x["thinki"] = th.replace("Classify the TEST case below.", INSTR + "\n\nClassify the TEST case below.", 1) + THINK
    else:
        x["direct"] = t
        x["instr"] = t.replace("Q: ", INSTR + " Q: ", 1)
        core = t[:-3] if t.endswith(" A:") else t
        x["think"] = core + THINK
        x["thinki"] = core.replace("Q: ", INSTR + " Q: ", 1) + THINK
        if x["pop"] in ("traps", "cfright", "agree") and t.split(" Q: ")[0]:
            x["truth"] = t.split(" Q: ")[0] + TRUTH
    return x


def build(S, small=False):
    rng = random.Random(S); k = 2 if small else 30
    items = build_R(rng, k)
    assert all(items[x["twin"]]["twin_of"] == i for i, x in enumerate(items) if "twin" in x)
    items += build_N(rng, S, small)
    for f, B in (("K", build_K), ("D", build_D)):
        for pop, n in (("traps", 10 * k), ("agree", int(10 * k / 3)), ("plain", int(4 * k / 3))):   # K 20x15, D 15x20
            items += B(rng, pop, n)
    old = old_texts() if not small else set()
    n0 = len(items)
    # keep twins aligned: drop a trap and its twin together
    for i, x in enumerate(items):
        x["_i"] = i
    txt = {}
    drop = set()
    for x in items:
        if x["text"].replace("  ", " ") in old or x["text"] in txt:
            drop.add(x["_i"])
        txt.setdefault(x["text"], x["_i"])
    for x in items:
        if "twin" in x and (x["_i"] in drop or x["twin"] in drop):
            drop |= {x["_i"], x["twin"]}
    kept = [x for x in items if x["_i"] not in drop]
    new = {x["_i"]: j for j, x in enumerate(kept)}
    for x in kept:
        if "twin" in x:
            x["twin"] = new[x["twin"]]
        if "twin_of" in x:
            x["twin_of"] = new[x["twin_of"]]
    print(f"dropped {n0 - len(kept)} items (duplicates of earlier sets / repeats; twins dropped in pairs)")
    items = kept
    for i, x in enumerate(items):
        x["uid"] = i; x.setdefault("grp", x["text"]); prompts(x); x.pop("_i")
        assert "twin" not in x or items[x["twin"]]["twin_of"] == i
    return items


if __name__ == "__main__":
    from collections import Counter
    small = "--small" in sys.argv; S = int([a for a in sys.argv[1:] if not a.startswith("--")][0])
    assert small or S not in USED, "seed already used"
    items = build(S, small)
    out = os.path.join(HERE, "sets_smoke.json" if small else "sets_ta.json")
    json.dump(dict(seed=S, instr=INSTR, think=THINK, items=items), open(out, "w"), ensure_ascii=False)
    print(len(items), "items ->", out, dict(sorted(Counter((x["family"], x["pop"]) for x in items).items())))
