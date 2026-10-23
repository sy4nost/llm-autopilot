"""v3 broad 'fu / good' data: 8 kinds of traps where the past / the familiar pulls the wrong way.
Each item: prompt, key (good), fu (bad), cands (for scoring), fam, kind.
kinds: change (explicit), plain (ordinary words), decoy (a change that does not concern this question: good = usual answer),
       same (nothing changed: good = usual / past answer), noisy (rules: past decisions were wrong)
Families: rule, fact (from v1/v2), redefine, altered, format, schedule, story, math.
python3 -I make_fu3.py <out_dir>  -> train3.json (and test3.json, same families, other seed + other words where possible)
"""
import json, random, sys
sys.path.insert(0, "..")
from make_fu import rule_item, fact_item, math_item, TRAIN_W, TEST_W, TRAIN_N, TEST_N
from make_fu2 import rule_v2, fact_v2, R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN, F_PLAIN_TRAIN, F_DECOY_TRAIN, \
    F_DECOY_PLAIN_TRAIN

pick = lambda rng, xs: xs[rng.randrange(len(xs))]


def item(fam, kind, prompt, key, fu, cands=None):
    return dict(fam=fam, kind=kind, prompt=prompt, key=str(key), fu=str(fu), old=str(fu),
                cands=[str(c) for c in (cands or [key, fu])])


# ---------- redefine: a word or a sign gets a new meaning in this task
WORDS = [("dozen", 12), ("pair", 2), ("week", 7), ("score", 20), ("trio", 3), ("hand", 5), ("century", 100)]
WORDS_T = [("gross", 144), ("decade", 10), ("quartet", 4), ("fortnight", 14)]
OPS = [("+", "addition", lambda a, b: a + b), ("-", "subtraction", lambda a, b: a - b), ("*", "multiplication", lambda a, b: a * b)]
RD_CHANGE = ["In this puzzle, one {w} means {k} (not {r}).", "For this question only, count a {w} as {k}.",
             "Note: here the word '{w}' stands for {k} things."]
RD_DECOY = ["In this puzzle, one {w2} means {k} (not {r2}).", "Note: in a different game, a {w} is counted as {k}; here it is normal."]


def redefine(rng, kind, words):
    if rng.random() < 0.55:
        (w, r), (w2, r2) = rng.sample(words, 2)
        k = rng.choice([x for x in range(2, 30) if x != r])
        n = rng.randint(2, 6)
        q = f"How many things are in {n} {w}s? Answer with the number only."
        if kind in ("change", "plain"):
            t = RD_CHANGE[0] if kind == "change" else pick(rng, RD_CHANGE[1:])
            return item("redefine", kind, t.format(w=w, k=k, r=r) + "\n" + q, n * k, n * r)
        if kind == "decoy":
            t = pick(rng, RD_DECOY)
            return item("redefine", kind, t.format(w=w, w2=w2, k=k, r2=r2) + "\n" + q, n * r, n * k)
        return item("redefine", "same", q, n * r, n * k)
    (s, nm, f), (s2, nm2, f2) = rng.sample(OPS, 2)
    a, b = rng.randint(6, 15), rng.randint(2, 5)
    q = f"What is {a} {s} {b}? Answer with the number only."
    if kind in ("change", "plain"):
        t = f"In this task the sign '{s}' means {nm2}." if kind == "change" else f"Here, read '{s}' as {nm2}."
        return item("redefine", kind, t + "\n" + q, f2(a, b), f(a, b))
    if kind == "decoy":
        other = [o for o in OPS if o[0] != s][0]
        o3 = [o for o in OPS if o[0] not in (s, other[0])][0]
        t = f"In this task the sign '{other[0]}' means {o3[1]}."
        return item("redefine", kind, t + "\n" + q, f(a, b), f2(a, b))
    return item("redefine", "same", q, f(a, b), f2(a, b))


# ---------- altered: a familiar text is changed; report what is written, not what is remembered
SAYINGS = [("The early bird catches the", "worm", "train"), ("Actions speak louder than", "words", "music"),
           ("Every cloud has a silver", "lining", "spoon"), ("Don't count your chickens before they", "hatch", "sing"),
           ("When in Rome, do as the Romans", "do", "eat"), ("A picture is worth a thousand", "words", "coins"),
           ("Better late than", "never", "sorry"), ("All that glitters is not", "gold", "glass"),
           ("Practice makes", "perfect", "progress"), ("Rome wasn't built in a", "day", "week"),
           ("The pen is mightier than the", "sword", "spoon"), ("An apple a day keeps the doctor", "away", "busy"),
           ("Too many cooks spoil the", "broth", "party"), ("Curiosity killed the", "cat", "plan"),
           ("Beauty is in the eye of the", "beholder", "storm"), ("Two heads are better than", "one", "none"),
           ("Where there's smoke, there's", "fire", "steam"), ("Laughter is the best", "medicine", "music")]
SAYINGS_T = [("Honesty is the best", "policy", "habit"), ("Absence makes the heart grow", "fonder", "colder"),
             ("Fortune favors the", "bold", "lucky"), ("Knowledge is", "power", "money")]
SEQS = [["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], ["January", "February", "March", "April", "May"],
        ["one", "two", "three", "four", "five"], ["red", "orange", "yellow", "green", "blue"]]


def altered(rng, kind, sayings):
    if rng.random() < 0.65:
        s, real, alt = pick(rng, sayings)
        q = "What is the last word of the sentence, exactly as written? Answer with one word."
        if kind in ("change", "plain"):
            lead = "Read carefully: " if kind == "change" else ""
            return item("altered", kind, f'{lead}Sentence: "{s} {alt}."\n{q}', alt, real)
        if kind == "decoy":
            s2, r2, a2 = pick(rng, [x for x in sayings if x[0] != s])
            return item("altered", kind, f'(Another sentence was changed: "{s2} {a2}.")\nSentence: "{s} {real}."\n{q}', real, alt)
        return item("altered", "same", f'Sentence: "{s} {real}."\n{q}', real, alt)
    seq = pick(rng, SEQS)
    i = rng.randint(1, 4)
    q = f"What is item number {i + 1} in this list, exactly as written? Answer with one word."
    if kind in ("change", "plain", "decoy"):
        alt = pick(rng, [x for s2 in SEQS for x in s2 if x not in seq])
        lst = seq[:i] + [alt] + seq[i + 1:]
        if kind == "decoy":
            j = rng.choice([x for x in range(5) if x != i])
            lst = seq[:j] + [alt] + seq[j + 1:]
            return item("altered", kind, "List: " + ", ".join(lst) + "\n" + q, seq[i], alt)
        return item("altered", kind, "List: " + ", ".join(lst) + "\n" + q, alt, seq[i])
    return item("altered", "same", "List: " + ", ".join(seq) + "\n" + q, seq[i], pick(rng, [x for x in seq if x != seq[i]]))


# ---------- format: earlier answers used one format, now the instruction changed
CAPS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Spain", "Madrid"),
        ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"), ("Cuba", "Havana"), ("Greece", "Athens"),
        ("Chile", "Santiago"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Thailand", "Bangkok")]
FMT = [("UPPERCASE", str.upper), ("lowercase", str.lower)]


def fmt_item(rng, kind):
    hist = rng.sample(CAPS, 7)
    (n1, f1), (n2, f2) = rng.sample(FMT, 2)
    test = hist[0]
    rows = [f"Q: capital of {c}? A: {f1(x)}" for c, x in hist[1:]] + [f"Q: capital of {test[0]}? A: {f1(test[1])}"]
    rng.shuffle(rows)
    q = f"Q: capital of {test[0]}? A:"
    note = {"change": f"NEW INSTRUCTION: from now on write answers in {n2}.",
            "plain": f"Please switch to {n2} for the answers from here on.",
            "decoy": f"NEW INSTRUCTION for the other assistant (not you): write answers in {n2}.",
            "same": ""}[kind]
    p = "Earlier answers:\n" + "\n".join(rows) + "\n\n" + (note + "\n\n" if note else "") + "Answer the next one.\n" + q
    good = f2(test[1]) if kind in ("change", "plain") else f1(test[1])
    bad = f1(test[1]) if kind in ("change", "plain") else f2(test[1])
    return item("format", kind, p, good, bad)


# ---------- schedule: usual times vs this week's change
PLACES = ["library", "gym", "pool", "bakery", "museum", "clinic"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def schedule(rng, kind):
    place = pick(rng, PLACES)
    h = {d: rng.choice([7, 8, 9, 10]) for d in DAYS}
    day = pick(rng, DAYS)
    h2 = rng.choice([x for x in (11, 12, 13, 6) if x != h[day]])
    visits = [f"Visit on {d}: it opened at {h[d]}." for d in rng.choices(DAYS, k=8)] + [f"Visit on {day}: it opened at {h[day]}."] * 2
    rng.shuffle(visits)
    other = pick(rng, [d for d in DAYS if d != day])
    note = {"change": f"NOTICE: this week the {place} opens at {h2} on {day}.",
            "plain": f"Heads-up from the staff: on {day} this week we start at {h2}.",
            "decoy": f"NOTICE: this week the {place} opens at {h2} on {other}.",
            "same": ""}[kind]
    p = (f"Past visits to the {place}:\n" + "\n".join(visits) + "\n\n" + (note + "\n\n" if note else "")
         + f"At what hour does the {place} open on {day} this week? Answer with the number only.")
    good, bad = (h2, h[day]) if kind in ("change", "plain") else (h[day], h2)
    return item("schedule", kind, p, good, bad)


# ---------- story: the story world has its own facts that override common knowledge
WORLD = [("iron floats on water", "Mia drops an iron key into a lake. Does the key float?", "Yes", "No"),
         ("the sun rises in the west", "Leo faces the sunrise. Is he facing west?", "Yes", "No"),
         ("snow is hot", "Ana touches fresh snow. Does it feel hot?", "Yes", "No"),
         ("fish breathe air and drown in water", "A fish is put in a full bucket of water. Is it safe?", "No", "Yes"),
         ("ice sinks in water", "Tom puts an ice cube in his glass of water. Does it sink?", "Yes", "No"),
         ("plants grow best in total darkness", "Sara wants her plant to grow. Should she keep it in the dark?", "Yes", "No"),
         ("dogs can fly", "Rex the dog wants to get onto the roof. Can he fly there?", "Yes", "No"),
         ("glass is unbreakable", "Ben drops a glass cup on stone. Does it break?", "No", "Yes"),
         ("water boils at 40 degrees", "Kim heats water to 50 degrees. Does it boil?", "Yes", "No"),
         ("the moon is made of cheese", "An astronaut takes a bite of the moon. Is it cheese?", "Yes", "No"),
         ("grass is blue", "Eva looks at a lawn. Is the grass blue?", "Yes", "No"),
         ("cats are afraid of mice", "A mouse enters the room with a cat. Is the cat scared?", "Yes", "No")]


def story(rng, kind):
    w = pick(rng, WORLD)
    w2 = pick(rng, [x for x in WORLD if x != w])
    if kind in ("change", "plain"):
        lead = f"In this story, {w[0]}." if kind == "change" else f"Remember that where this story happens, {w[0]}."
        return item("story", kind, lead + "\n" + w[1] + " Answer Yes or No.", w[2], w[3], ["Yes", "No"])
    if kind == "decoy":
        return item("story", kind, f"In this story, {w2[0]}.\n" + w[1] + " Answer Yes or No.", w[3], w[2], ["Yes", "No"])
    return item("story", "same", w[1] + " Answer Yes or No.", w[3], w[2], ["Yes", "No"])


def math3(rng, kind):
    it = math_item(rng, "change" if kind in ("change", "plain") else ("same" if kind == "same" else "subtle"))
    it["fu"] = it["old"] if kind != "same" else it["cands"][1]
    if kind == "decoy":   # 'subtle' = a number silently changed: good = new answer, fu = copied; keep kind name
        it["kind"] = "subtle"
    return it


def build(seed, n_scale, test=False):
    rng = random.Random(seed)
    W, N, SY, WD = (TEST_W, TEST_N, SAYINGS_T, WORDS_T + WORDS[:3]) if test else (TRAIN_W, TRAIN_N, SAYINGS, WORDS)
    out = []
    K = lambda n: max(1, int(n * n_scale))
    out += [rule_item(rng, W, "change") for _ in range(K(110))]
    out += [rule_item(rng, W, "same") for _ in range(K(80))]
    out += [rule_item(rng, W, "noisy") for _ in range(K(70))]
    out += [rule_v2(rng, W, "plain", R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN) for _ in range(K(110))]
    out += [rule_v2(rng, W, "decoy", R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN) for _ in range(K(100))]
    out += [fact_item(rng, N, "change") for _ in range(K(60))]
    out += [fact_item(rng, N, "same") for _ in range(K(50))]
    out += [fact_v2(rng, N, "plain", F_PLAIN_TRAIN, F_DECOY_TRAIN, F_DECOY_PLAIN_TRAIN) for _ in range(K(70))]
    out += [fact_v2(rng, N, "decoy", F_PLAIN_TRAIN, F_DECOY_TRAIN, F_DECOY_PLAIN_TRAIN) for _ in range(K(70))]
    for fn, args, counts in ((redefine, (WD,), (130, 90, 90, 90)), (altered, (SY,), (110, 80, 80, 80)),
                             (lambda r, k: fmt_item(r, k), (), (100, 80, 80, 70)),
                             (lambda r, k: schedule(r, k), (), (100, 80, 80, 70)),
                             (lambda r, k: story(r, k), (), (110, 80, 90, 80)),
                             (lambda r, k: math3(r, k), (), (130, 0, 80, 90))):
        for kind, n in zip(("change", "plain", "decoy", "same"), counts):
            out += [fn(rng, kind, *args) for _ in range(K(n))]
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    tr = build(20261019, 1.0)
    te = build(20261020, 0.25, test=True)
    json.dump(tr, open(sys.argv[1] + "/train3.json", "w"), indent=1)
    json.dump(te, open(sys.argv[1] + "/test3.json", "w"), indent=1)
    from collections import Counter
    print("train3", len(tr), sorted(Counter((x["fam"], x["kind"]) for x in tr).items()))
    print("test3", len(te))
    bad = [x for x in tr + te if x["key"] == x["fu"]]
    print("key==fu:", len(bad), [ (b["fam"], b["kind"]) for b in bad[:5]])
