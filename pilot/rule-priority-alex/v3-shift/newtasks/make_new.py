"""Unfamiliar task types for the frozen "lamp -> louder new rule" test (see PREREG.md). Two types, never used before:
  robot   a robot command language: made-up words mean actions; the update changes the meaning of one word.
  points  coloured tokens are worth points; answer = total (one digit); the update changes the value of one colour.
Per set: probes T1, T2 (test case touched by the change) and S1, S2 (not touched, old answer = new answer).
Arms (one test case per prompt):
  A60      60 old examples (test case appears twice) + update note + test                       key: new
  E_copy   PERIOD 1 (same 60) + update + PERIOD 2: 10 examples under the new rule incl. the test   key: new
  E_nocopy same, but the test-case slot in PERIOD 2 holds a stand-in not touched by the change      key: new
  N_same   PERIOD 1 + PERIOD 2 under the old rule incl. the test, no update                       key: old
  N60      60 old examples + test, no update                                                       key: old
  R60      60 old examples + a note saying the rule was reviewed and stays the same + test         key: old
  F0       the new rule written directly into the rules, no examples, no note (ceiling)            key: new
python3 -I make_new.py SEED -> sets_new.json
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_SETS = 20
WORDS = ["blick", "zorp", "mibu", "trell", "quon", "vesk", "dramo", "plif", "skeb", "nuro", "gaft", "hilo", "wumb", "toxa"]
ACTIONS = ["LEFT", "RIGHT", "JUMP", "STOP", "WAIT", "TURN", "BACK", "DUCK"]
COLOURS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
QUERY = "Answer for the TEST case below. Do not explain. Return only the answer.\n\nTEST CASE: {case}\n\nReturn only the answer."


# ---------------- robot ----------------
def robot_set(rng):
    w = rng.sample(WORDS, 5)
    acts = rng.sample(ACTIONS, 6)        # 5 meanings + 1 spare action used as the new meaning
    s = dict(kind="robot", words=w, mean=dict(zip(w, acts[:5])), ch=w[0], new_mean=acts[5])
    return s


def robot_rules(s, new):
    m = dict(s["mean"])
    if new:
        m[s["ch"]] = s["new_mean"]
    lines = "\n".join(f"- the word '{x}' means {m[x]}" for x in s["words"])
    return ("RESEARCH TASK: ROBOT COMMAND LANGUAGE\n\nUse ONLY the following fictional command language. A robot receives one "
            f"command word at a time and performs one action.\n\nDICTIONARY:\n{lines}\n\n"
            f"Translate each command word into the robot's action. Answer with the action in capital letters.\n\n")


def robot_ans(s, c, new):
    return s["new_mean"] if (new and c == s["ch"]) else s["mean"][c]


def robot_case(s, c):
    return f"command '{c}'"


def robot_probes(s, rng):
    others = s["words"][1:]
    pr = {"T1": s["ch"], "T2": s["ch"], "S1": others[0], "S2": others[1]}
    return pr


def robot_hist(s, rng, n, test):
    # changed word in ~1/3 of examples; the test case appears exactly twice
    cs = [test, test]
    while len(cs) < n:
        cs.append(s["ch"] if rng.random() < 0.3 else rng.choice(s["words"][1:]))
    rng.shuffle(cs)
    return cs


def robot_p2(s, rng, test):
    cs = ["TEST", s["ch"], s["ch"]] + [rng.choice(s["words"][1:]) for _ in range(7)]
    rng.shuffle(cs)
    stand = rng.choice([x for x in s["words"][1:]])
    return cs, stand


def robot_labels(s, c, rng):
    old, new = robot_ans(s, c, False), robot_ans(s, c, True)
    pool = [a for a in ACTIONS if a not in (old, new)]
    labs = [old] + ([new] if new != old else []) + rng.sample(pool, 3 - (1 + (new != old)))
    rng.shuffle(labs)
    return labs


# ---------------- points ----------------
def points_set(rng):
    col = rng.sample(COLOURS, 3)
    val = rng.sample([1, 2, 3], 3)
    ch = col[rng.randrange(3)]
    new_val = rng.choice([v for v in (1, 2, 3, 4) if v != dict(zip(col, val))[ch]])
    return dict(kind="points", cols=col, val=dict(zip(col, val)), ch=ch, new_val=new_val)


def points_rules(s, new):
    v = dict(s["val"])
    if new:
        v[s["ch"]] = s["new_val"]
    lines = "\n".join(f"- a {c} token is worth {v[c]} point{'s' if v[c] > 1 else ''}" for c in s["cols"])
    return ("RESEARCH TASK: TOKEN POINTS\n\nUse ONLY the following fictional scoring system. A case is a small pile of coloured "
            f"tokens.\n\nVALUES:\n{lines}\n\nThe score of a case is the sum of the values of its tokens. Answer with the score as a "
            "single number.\n\n")


def points_ans(s, c, new):
    v = dict(s["val"])
    if new:
        v[s["ch"]] = s["new_val"]
    return str(sum(v[col] * k for col, k in c))


def points_case(s, c):
    return ", ".join(f"{k} {col}" for col, k in c)


def points_rand(s, rng, with_ch):
    for _ in range(1000):
        cols = rng.sample(s["cols"], rng.choice((1, 2)))
        if with_ch and s["ch"] not in cols:
            cols = [s["ch"]] + cols[:1]
        if not with_ch and s["ch"] in cols:
            continue
        c = tuple((col, rng.choice((1, 2))) for col in sorted(set(cols), key=s["cols"].index))
        a, b = int(points_ans(s, c, False)), int(points_ans(s, c, True))
        if 1 <= min(a, b) and max(a, b) <= 9:
            return c
    raise RuntimeError("no case")


def points_probes(s, rng):
    t1 = points_rand(s, rng, True)
    t2 = next(c for c in iter(lambda: points_rand(s, rng, True), None) if c != t1)
    s1 = points_rand(s, rng, False)
    s2 = next(c for c in iter(lambda: points_rand(s, rng, False), None) if c != s1)
    return {"T1": t1, "T2": t2, "S1": s1, "S2": s2}


def points_hist(s, rng, n, test):
    cs = [test, test]
    while len(cs) < n:
        cs.append(points_rand(s, rng, rng.random() < 0.35))
    rng.shuffle(cs)
    return cs


def points_p2(s, rng, test):
    cs = ["TEST", points_rand(s, rng, True), points_rand(s, rng, True)] + [points_rand(s, rng, False) for _ in range(7)]
    rng.shuffle(cs)
    return cs, points_rand(s, rng, False)


def points_labels(s, c, rng):
    old, new = points_ans(s, c, False), points_ans(s, c, True)
    pool = [str(d) for d in range(1, 10) if str(d) not in (old, new)]
    labs = [old] + ([new] if new != old else []) + rng.sample(pool, 3 - (1 + (new != old)))
    rng.shuffle(labs)
    return labs


T = {"robot": dict(mk=robot_set, rules=robot_rules, ans=robot_ans, case=robot_case, probes=robot_probes, hist=robot_hist,
                   p2=robot_p2, labels=robot_labels,
                   update=lambda s: f"IMPORTANT UPDATE: From this point on, the word '{s['ch']}' means {s['new_mean']}. All other words keep their meaning.\n\n",
                   review=lambda s: f"IMPORTANT UPDATE: The meaning of the word '{s['ch']}' has been reviewed and stays the same: it means {s['mean'][s['ch']]}. All other words keep their meaning.\n\n"),
     "points": dict(mk=points_set, rules=points_rules, ans=points_ans, case=points_case, probes=points_probes, hist=points_hist,
                    p2=points_p2, labels=points_labels,
                    update=lambda s: f"IMPORTANT UPDATE: From this point on, a {s['ch']} token is worth {s['new_val']} point{'s' if s['new_val'] > 1 else ''}. All other values stay the same.\n\n",
                    review=lambda s: f"IMPORTANT UPDATE: The value of a {s['ch']} token has been reviewed and stays the same: {s['val'][s['ch']]} point{'s' if s['val'][s['ch']] > 1 else ''}. All other values stay the same.\n\n")}


def rows(t, s, cs, new, start):
    return "\n".join(f"Example {j:02d}: {t['case'](s, c)} -> {t['ans'](s, c, new)}" for j, c in enumerate(cs, start))


def build(seed):
    rng = random.Random(seed)
    items = []
    for kind, t in T.items():
        for i in range(N_SETS):
            s = t["mk"](rng)
            for pid, c in t["probes"](s, rng).items():
                labs = t["labels"](s, c, rng)
                old, new = t["ans"](s, c, False), t["ans"](s, c, True)
                p1 = t["hist"](s, rng, 60, c)
                slots, stand = t["p2"](s, rng, c)
                q = QUERY.format(case=t["case"](s, c))
                head = t["rules"](s, False)
                intro1 = "Below are 60 solved examples. They show cases and their correct answers at that time.\n\n"
                intro_p1 = "Below are 60 solved examples from PERIOD 1. They show cases and their correct answers at that time.\n\n"
                P = {}
                P["A60"] = (head + intro1 + rows(t, s, p1, False, 1) + "\n\n" + t["update"](s) + q, new)
                for arm, copy, changed in (("E_copy", True, True), ("E_nocopy", False, True), ("N_same", True, False)):
                    p2 = [(c if copy else stand) if x == "TEST" else x for x in slots]
                    P[arm] = (head + intro_p1 + rows(t, s, p1, False, 1) + "\n\n" + (t["update"](s) if changed else "")
                              + "Below are 10 solved examples from PERIOD 2" + (" (after the update)" if changed else "") + ".\n\n"
                              + rows(t, s, p2, changed, 61) + "\n\n" + q, new if changed else old)
                P["N60"] = (head + intro1 + rows(t, s, p1, False, 1) + "\n\n" + q, old)
                P["R60"] = (head + intro1 + rows(t, s, p1, False, 1) + "\n\n" + t["review"](s) + q, old)
                P["F0"] = (t["rules"](s, True) + q, new)
                for arm, (prompt, k) in P.items():
                    items.append(dict(task=kind, set=i, arm=arm, probe=pid, prompt=prompt, labels=labs, key=k, old=old,
                                      changed=(old != new), rule=s, case=c, p1=p1,
                                      p2=None if arm not in ("E_copy", "E_nocopy", "N_same") else
                                      [(c if arm != "E_nocopy" else stand) if x == "TEST" else x for x in slots]))
    return items


if __name__ == "__main__":
    seed = int(sys.argv[1])
    items = build(seed)
    out = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "sets_new.json")
    json.dump(dict(seed=seed, items=items), open(out, "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts ->", out, "| max chars", max(len(x["prompt"]) for x in items))
