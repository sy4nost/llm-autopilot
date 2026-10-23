"""Items for the ask-twice checker (PLAN.md v2).
  python3 -I make_at.py --dev                 -> sets_dev.json   (seed 12345, every variant, traps + plain, for the variant choice)
  python3 -I make_at.py S [dev_choice.json]   -> sets_at.json    (main: chosen variant per family, all populations)
Every item: family, pop, variant, cell (= hook), grp (distinct text id), text (normal), disg (module output), para, instr,
truth (S2 prompt), cands (3 options), key (index of gold), habit (index of the real-world option or None)."""
import json, os, re, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
USED = {12345, 25, 32, 185, 20261018, 20261020, 0, 90} | set(range(88, 99))
INSTR = "Follow the facts stated in this task exactly, even if they contradict what you know."
TRUTH = "\n\nIs the first sentence of this task true in the real world? Reply yes or no."
FILL = ["The weather was mild.", "Nobody was in a hurry.", "A radio was playing quietly in the next room.",
        "The table was covered with old papers.", "Somebody laughed in the corridor.", "The coffee had gone cold.",
        "Outside, a bus stopped and drove on.", "The lamp on the desk flickered once."]
NAMES = ["Ana", "Tom", "Marie", "Omar", "Lena", "Ravi"]
LEADS = ["In this world,", "In this story,", "In this game,"]
NONCE = {"U": ["vorn", "drell", "quisk", "tamber"], "K": ["Zelvar", "Drovany", "Quenland", "Jorvesh"],
         "D": ["zib", "florp", "kesh", "wumble"]}
# module: one frozen regex per family finds the hook word in the first sentence (template-aware, disclosed in PLAN)
HOOK_RE = {"U": r"\ban? (\w+) has \d+ \w+\.", "K": r"capital of (\w+) is \w[\w ]*\.", "D": r"\ban? '(\w+)' has \d+ \w+\."}

# ---------- U units: hook, plural, sub unit, real value, knowledge question (gold, wrong1, wrong2)
U = [("week", "weeks", "days", 7, ("How many weeks are in a fortnight?", "2", "3", "4")),
     ("day", "days", "hours", 24, ("How many days are in a week?", "7", "5", "10")),
     ("hour", "hours", "minutes", 60, ("How many hours are in a day?", "24", "12", "20")),
     ("minute", "minutes", "seconds", 60, ("How many minutes are in an hour?", "60", "100", "30")),
     ("year", "years", "months", 12, ("How many years are in a decade?", "10", "12", "5")),
     ("decade", "decades", "years", 10, ("How many decades are in a century?", "10", "5", "12")),
     ("century", "centuries", "years", 100, ("How many centuries are in a millennium?", "10", "100", "5")),
     ("dozen", "dozens", "items", 12, ("How many dozens are in a gross?", "12", "10", "20")),
     ("foot", "feet", "inches", 12, ("How many feet are in a yard?", "3", "12", "10")),
     ("yard", "yards", "feet", 3, ("How many yards are in a mile?", "1760", "1000", "5280"))]
# ---------- K capitals: country, adjective, real capital, other city 1, other city 2, language, two wrong languages
K = [("France", "French", "Paris", "Lyon", "Marseille", "French", "German", "Italian"),
     ("Germany", "German", "Berlin", "Munich", "Hamburg", "German", "Dutch", "Polish"),
     ("Italy", "Italian", "Rome", "Milan", "Naples", "Italian", "Spanish", "Greek"),
     ("Spain", "Spanish", "Madrid", "Barcelona", "Seville", "Spanish", "Portuguese", "Italian"),
     ("Japan", "Japanese", "Tokyo", "Osaka", "Kyoto", "Japanese", "Chinese", "Korean"),
     ("Russia", "Russian", "Moscow", "Kazan", "Novosibirsk", "Russian", "Polish", "Turkish"),
     ("Egypt", "Egyptian", "Cairo", "Alexandria", "Luxor", "Arabic", "Turkish", "French"),
     ("Brazil", "Brazilian", "Brasilia", "Recife", "Salvador", "Portuguese", "Spanish", "French"),
     ("Canada", "Canadian", "Ottawa", "Toronto", "Montreal", "English", "German", "Dutch"),
     ("Australia", "Australian", "Canberra", "Sydney", "Melbourne", "English", "German", "Spanish"),
     ("China", "Chinese", "Beijing", "Shanghai", "Wuhan", "Chinese", "Japanese", "Korean"),
     ("Turkey", "Turkish", "Ankara", "Istanbul", "Izmir", "Turkish", "Greek", "Arabic"),
     ("Poland", "Polish", "Warsaw", "Krakow", "Gdansk", "Polish", "Russian", "German"),
     ("Greece", "Greek", "Athens", "Thessaloniki", "Patras", "Greek", "Italian", "Turkish"),
     ("Sweden", "Swedish", "Stockholm", "Gothenburg", "Malmo", "Swedish", "German", "Polish"),
     ("Portugal", "Portuguese", "Lisbon", "Porto", "Braga", "Portuguese", "Spanish", "French"),
     ("Kenya", "Kenyan", "Nairobi", "Mombasa", "Kisumu", "Swahili", "Arabic", "French"),
     ("Norway", "Norwegian", "Oslo", "Bergen", "Trondheim", "Norwegian", "German", "Russian"),
     ("Austria", "Austrian", "Vienna", "Salzburg", "Graz", "German", "Italian", "Polish"),
     ("Argentina", "Argentine", "Buenos Aires", "Cordoba", "Rosario", "Spanish", "Portuguese", "Italian")]
# ---------- D defined words: hook, plural, counted part, real count, category, two wrong categories
D = [("triangle", "triangles", "sides", 3, "a shape", "an animal", "a vehicle"),
     ("square", "squares", "sides", 4, "a shape", "an animal", "a food"),
     ("hexagon", "hexagons", "sides", 6, "a shape", "a vehicle", "a food"),
     ("cube", "cubes", "faces", 6, "a shape", "an animal", "a vehicle"),
     ("spider", "spiders", "legs", 8, "an animal", "a shape", "a vehicle"),
     ("ant", "ants", "legs", 6, "an animal", "a shape", "a food"),
     ("dog", "dogs", "legs", 4, "an animal", "a vehicle", "a shape"),
     ("octopus", "octopuses", "arms", 8, "an animal", "a food", "a shape"),
     ("bicycle", "bicycles", "wheels", 2, "a vehicle", "an animal", "a shape"),
     ("tricycle", "tricycles", "wheels", 3, "a vehicle", "a food", "an animal"),
     ("hand", "hands", "fingers", 5, "a body part", "a vehicle", "a shape")]


def fake_val(rng, real):
    while True:
        v = rng.choice([real - 2, real - 1, real + 1, real + 2, real + 3, real * 2, max(2, real // 2)])
        if v > 1 and v != real:
            return v


def first_free(cands, taken):
    return next(c for c in cands if c not in taken)


def three(rng, gold, habit, dist):
    c = [gold, habit, dist]
    assert len(set(c)) == 3, c
    order = list(range(3)); rng.shuffle(order)
    cands = [c[i] for i in order]
    return cands, cands.index(gold), (cands.index(habit) if habit is not None else None)


def three_k(rng, gold, w1, w2):
    cands, k, _ = three(rng, gold, w1, w2)
    return cands, k, None


def bal(rng, L, n):
    """n picks cycling through L in shuffled rounds (balanced cells)"""
    out = []
    while len(out) < n:
        r = list(L); rng.shuffle(r); out += r
    return out[:n]


def filler(rng, k=3):
    return " " + " ".join(rng.sample(FILL, k))


def build_U(rng, pop, variant, n_items):
    out = []
    for row in bal(rng, U, n_items):
        hk, pl, sub, real, kn = row
        lead = rng.choice(LEADS); n = rng.randint(2, 6); m = rng.randint(2, max(2, min(real - 1, 9)))
        fake = fake_val(rng, real)
        art = "an" if hk[0] in "aeiouh" and hk != "hand" and hk not in ("year",) else "a"
        art = "an" if hk == "hour" else art
        val = real if pop == "agree" else fake
        ctx = f"{lead} {art} {hk} has {val} {sub}." if pop != "plain" else ""
        if pop in ("traps", "agree", "plain"):
            q, g, h, d = q_U(rng, variant, hk, pl, sub, n, m, fake if pop == "traps" else real, real)
            if pop != "traps":
                h = None
            cands, key, hab = three(rng, g, h if h is not None else d[1], d[0])
            if pop != "traps":
                hab = None
        elif pop == "knowledge":
            q, g, w1, w2 = kn; cands, key, hab = three_k(rng, g, w1, w2)
        else:  # irrelevant: question about another hook, real answer
            hk2, pl2, sub2, real2, _ = rng.choice([x for x in U if x[0] != hk and x[2] not in (hk, pl)])
            q = f"How many {sub2} are in {n} {pl2}?"
            g2 = n * real2; d1 = first_free([g2 + n, g2 + 1], (g2,)); d2 = first_free([(n + 1) * real2, g2 + 2 * n, g2 + 2], (g2, d1))
            cands, key, hab = three_k(rng, str(g2), str(d1), str(d2))
        text = (ctx + filler(rng) if variant == 4 and ctx and pop in ("traps", "agree") else ctx)
        out.append(dict(family="U", pop=pop, variant=variant, cell=hk, ctx=ctx, text=(text + " Q: " if text else "Q: ") + q + " A:",
                        cands=cands, key=key, habit=hab))
    return out


def q_U(rng, variant, hk, pl, sub, n, m, val, real):
    """returns question, gold, habit (real-world answer), (distractor, spare)"""
    name = rng.choice(NAMES)
    if variant == 1:
        q = f"How many {sub} are in {n} {pl}?"; g, h = n * val, n * real
    elif variant == 2:
        q = f"How many {sub} are in {n} {pl} and {m} {sub}?"; g, h = n * val + m, n * real + m
    else:
        q = f"{name} has {n} {pl} to deal with. How many {sub} is that in total?"; g, h = n * val, n * real
    d = first_free([(n + 1) * val, n * val + n * real, g + h + 1], (g, h))
    spare = first_free([g + n, g + 2 * n, g + h + 2], (g, h, d))
    return q, str(g), str(h), (str(d), str(spare))


def build_K(rng, pop, variant, n_items):
    out = []
    for row in bal(rng, K, n_items):
        c, adj, cap, o1, o2, lang, wl1, wl2 = row
        lead = rng.choice(LEADS); fake = rng.choice([o1, o2]); third = o2 if fake == o1 else o1
        val = cap if pop == "agree" else fake
        ctx = f"{lead} the capital of {c} is {val}." if pop != "plain" else ""
        name = rng.choice(NAMES); how = rng.choice(["flew", "took a train", "drove"])
        if pop in ("traps", "agree", "plain"):
            g = fake if pop == "traps" else cap
            if variant == 1:
                q = rng.choice([f"What is the first letter of the capital of {c}?", f"The capital of {c} starts with which letter?"])
                pool = [x[0] for x in (g, cap, third) if x[0] != g[0]]
                L = sorted({chr(i) for i in range(65, 91)} - {g[0], cap[0]} - {x[0] for x in NONCE["K"]})
                h = cap[0] if pop == "traps" else None
                dist = third[0] if third[0] not in (g[0], cap[0]) else rng.choice(L)
                if pop != "traps":
                    h = rng.choice([x for x in L if x != dist])
                cands, key, hab = three(rng, g[0], h, dist)
            else:
                q = f"{name} {how} to the {adj} capital. Which city did {name} arrive in?"
                h = cap if pop == "traps" else third
                dist = third if pop == "traps" else (o1 if fake != o1 else o2)
                if pop != "traps":
                    dist = o1 if third != o1 else o2
                cands, key, hab = three(rng, g, h, dist)
            if pop != "traps":
                hab = None
        elif pop == "knowledge":
            q = f"What language is mainly spoken in {c}?"; cands, key, hab = three_k(rng, lang, wl1, wl2)
        else:
            c2 = rng.choice([x for x in K if x[0] != c and x[1] != adj and x[5] != adj])
            q = f"{name} {how} to the {c2[1]} capital. Which city did {name} arrive in?" if variant > 1 else f"What is the capital of {c2[0]}?"
            cands, key, hab = three_k(rng, c2[2], c2[3], c2[4])
        text = (ctx + filler(rng) if variant == 3 and ctx and pop in ("traps", "agree") else ctx)
        out.append(dict(family="K", pop=pop, variant=variant, cell=c, ctx=ctx, text=(text + " Q: " if text else "Q: ") + q + " A:",
                        cands=cands, key=key, habit=hab))
    return out


def build_D(rng, pop, variant, n_items):
    out = []
    for row in bal(rng, D, n_items):
        hk, pl, part, real, cat, wc1, wc2 = row
        lead = rng.choice(LEADS); n = rng.randint(2, 6); fake = fake_val(rng, real); name = rng.choice(NAMES)
        art = "an" if hk[0] in "aeiou" else "a"
        val = real if pop == "agree" else fake
        ctx = f"{lead} {art} '{hk}' has {val} {part}." if pop != "plain" else ""
        if pop in ("traps", "agree", "plain"):
            v = fake if pop == "traps" else real
            if variant == 1:
                q = f"How many {part} does {art} {hk} have?"; g, h = v, real
            elif variant == 2:
                q = f"How many {part} do {n} {pl} have in total?"; g, h = n * v, n * real
            else:
                q = f"{name} counted {n} {pl}. How many {part} did {name} count in total?"; g, h = n * v, n * real
            mult = 1 if variant == 1 else n
            if pop != "traps":
                h = g + mult
            d = first_free([(mult + 1) * v if variant > 1 else v + real + 1, g + h, g + h + 1], (g, h))
            cands, key, hab = three(rng, str(g), str(h), str(d))
            if pop != "traps":
                hab = None
        elif pop == "knowledge":
            q = f"Is {art} {hk} a shape, an animal, a vehicle, a body part or a food? Answer with one of them."
            cands, key, hab = three_k(rng, cat, wc1, wc2)
        else:
            hk2, pl2, part2, real2 = rng.choice([x for x in D if x[0] != hk and x[2] != part])[:4]
            q = f"How many {part2} do {n} {pl2} have in total?"
            g2 = n * real2; d1 = first_free([g2 + n, g2 + 1], (g2,)); d2 = first_free([(n + 1) * real2, g2 + 2 * n, g2 + 2], (g2, d1))
            cands, key, hab = three_k(rng, str(g2), str(d1), str(d2))
        text = (ctx + filler(rng) if variant == 4 and ctx and pop in ("traps", "agree") else ctx)
        out.append(dict(family="D", pop=pop, variant=variant, cell=hk, ctx=ctx, text=(text + " Q: " if text else "Q: ") + q + " A:",
                        cands=cands, key=key, habit=hab))
    return out


# ---------- module, rivals
U_FORMS = {h: [(pl, True), (h, False)] for h, pl, *_ in U}
K_FORMS = {c: [(c, False), (adj, "adj")] for c, adj, *_ in K}
D_FORMS = {h: [(pl, True), (h, False)] for h, pl, *_ in D}


def nonce_for(fam, text):
    import hashlib
    return NONCE[fam][int(hashlib.sha256(text.encode()).hexdigest(), 16) % 4]


def disguise(fam, text):
    """sees only the text: find the hook in the first sentence with the family regex; replace hook forms everywhere"""
    m = re.search(HOOK_RE[fam], text.split(" Q: ")[0])
    if not m:
        return text
    hook = m.group(1)
    forms = {"U": U_FORMS, "K": K_FORMS, "D": D_FORMS}[fam].get(hook)
    if forms is None:
        return text
    nz = nonce_for(fam, text); out = text
    for f, kind in forms:
        rep = nz + "ian" if kind == "adj" else (nz + ("es" if nz.endswith(("s", "sh")) else "s")) if kind is True else nz
        out = re.sub(r"\b%s\b" % re.escape(f), rep, out)
    art = "an" if nz[0].lower() in "aeiou" else "a"
    return re.sub(r"\b(an?) ('?)%s\b" % re.escape(nz), lambda m: art + " " + m.group(2) + nz, out)


def paraphrase(it):
    q = it["text"].split("Q: ", 1)[1].rsplit(" A:", 1)[0]
    pre = it["text"].split("Q: ", 1)[0].strip()
    return (f"Background: {pre}\n" if pre else "") + f"Question: {q}\nAnswer:"


def finish(items, rng):
    for k, it in enumerate(items):
        it["disg"] = disguise(it["family"], it["text"])
        it["para"] = paraphrase(it)
        it["instr"] = it["text"].replace("Q: ", INSTR + " Q: ", 1)
        it["truth"] = it["text"].rsplit(" Q: ", 1)[0] + TRUTH if it["ctx"] else None
        it["grp"] = it["text"]
        it["nonce"] = nonce_for(it["family"], it["text"])
        q = it["text"].split("Q: ", 1)[1]
        forms = {"U": U_FORMS, "K": K_FORMS, "D": D_FORMS}[it["family"]][it["cell"]]
        it["hook_in_q"] = any(re.search(r"\b%s\b" % re.escape(f), q) for f, _ in forms)
        assert it["hook_in_q"] == (it["pop"] != "irrelevant"), it["text"]
    return items


BUILD = {"U": build_U, "K": build_K, "D": build_D}
VARIANTS = {"U": [1, 2, 3, 4], "K": [1, 2, 3], "D": [1, 2, 3, 4]}


def make_dev():
    rng = random.Random(12345); items = []
    for f in "UKD":
        for v in VARIANTS[f]:
            items += BUILD[f](rng, "traps", v, 120)
            items += BUILD[f](rng, "plain", v, 30)
    seen, keep = set(), []
    for x in items:
        if x["text"] not in seen:
            seen.add(x["text"]); keep.append(x)
    return finish(keep, rng)


def make_main(S, choice, small=False):
    assert small or S not in USED, "seed already used"
    rng = random.Random(S); items = []; k = 8 if small else 120
    for f, v in choice["variants"].items():
        items += BUILD[f](rng, "traps", v, 2 * k)
        for pop in ("agree", "knowledge", "irrelevant", "plain"):
            if pop == "knowledge" and f in ("U", "D"):
                continue
            items += BUILD[f](rng, pop, v, k)
    dev = {x["text"] for x in make_dev()} if not small else set()
    seen, keep, drop = set(), [], 0
    for x in items:
        if x["text"] in dev or x["text"] in seen:
            drop += 1; continue
        seen.add(x["text"]); keep.append(x)
    print("dropped as duplicates of dev or earlier main items:", drop)
    return finish(keep, rng)


if __name__ == "__main__":
    from collections import Counter
    if sys.argv[1] == "--dev":
        items = make_dev(); out = os.path.join(HERE, "sets_dev.json")
    else:
        small = "--smoke" in sys.argv; args = [a for a in sys.argv[1:] if not a.startswith("--")]
        S = int(args[0]); ch = json.load(open(os.path.join(HERE, args[1] if len(args) > 1 else "dev_choice.json")))
        items = make_main(S, ch, small); out = os.path.join(HERE, "sets_smoke.json" if small else "sets_at.json")
    for k, x in enumerate(items):
        x["uid"] = k
    json.dump(dict(items=items, instr=INSTR, truth=TRUTH), open(out, "w"), ensure_ascii=False)
    print(len(items), "items ->", out, dict(Counter((x["family"], x["pop"], x["variant"]) for x in items)))
