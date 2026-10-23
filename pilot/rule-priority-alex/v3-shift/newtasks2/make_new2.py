"""Run 2 of the frozen "lamp -> louder new rule" test (PREREG.md): unfamiliar task types where the change amends ONE PART of a
compound condition ("A and, at the same time, B"), the kind of change where 14B kept the old answer before.
  parcels  prose cases; priority rules choose a transport; Rule 2 'insured AND express -> AIR' becomes 'insured AND standard -> AIR'.
  bonus    coloured tokens worth points + a bonus 'contains X and, at the same time, a Y token: +3' that becomes '... NO Y token'.
Per set: probes T1, T2 (old Rule 2 / bonus fires, new does not: the answer changes) and S1, S2 (change does not touch them).
Arms as in run 1 (newtasks/make_new.py): A60, E_copy, E_nocopy, N_same, N60, R60, F0.
python3 -I make_new2.py SEED [out] -> sets_new2.json
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_SETS = 20
QUERY = "Answer for the TEST case below. Do not explain. Return only the answer.\n\nTEST CASE: {case}\n\nReturn only the answer."
LABS = [("TRUCK", "AIR", "SHIP", "VAN"), ("LORRY", "PLANE", "BOAT", "BIKE"), ("RAIL", "JET", "FERRY", "COURIER"),
        ("CART", "DRONE", "BARGE", "POST")]
COLOURS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
art = lambda w: "an" if w[0] in "aeiou" else "a"


# ---------------- parcels: case = (heavy, abroad, insured, express) ----------------
def parcels_set(rng):
    return dict(kind="parcels", lab=list(rng.choice(LABS)))


def parcels_r2(new):
    return "standard" if new else "express"


def parcels_rules(s, new):
    T, A, S, V = s["lab"]
    return ("RESEARCH TASK: PARCEL ROUTING\n\nUse ONLY the following fictional routing system. Apply its rules in the stated "
            "priority order, to every parcel independently.\n\n"
            f"RULE 1 (highest priority): A heavy parcel goes by {T}.\n"
            f"RULE 2: If Rule 1 does not apply, a parcel that is insured and, at the same time, {parcels_r2(new)} goes by {A}.\n"
            f"RULE 3: If Rules 1 and 2 do not apply, a parcel to an address abroad goes by {S}.\n"
            f"RULE 4: All remaining parcels go by {V}.\n\n"
            "Every parcel is described by its weight (light/heavy), its address (home/abroad), insurance (insured/not insured) "
            "and delivery (express/standard). Answer with the transport in capital letters.\n\n")


def parcels_ans(s, c, new):
    T, A, S, V = s["lab"]
    heavy, abroad, ins, exp = c
    if heavy: return T
    if ins and (exp if not new else not exp): return A
    return S if abroad else V


def parcels_case(s, c):
    heavy, abroad, ins, exp = c
    return (f"a {'heavy' if heavy else 'light'} parcel to {'an address abroad' if abroad else 'a home address'}, "
            f"{'insured' if ins else 'not insured'}, {'express' if exp else 'standard'} delivery")


def parcels_probes(s, rng):
    return {"T1": (0, 1, 1, 1), "T2": (0, 0, 1, 1), "S1": (1, rng.choice((0, 1)), rng.choice((0, 1)), rng.choice((0, 1))),
            "S2": (0, rng.choice((0, 1)), 0, rng.choice((0, 1)))}


def parcels_rand(rng):
    return (int(rng.random() < 0.2), rng.choice((0, 1)), rng.choice((0, 1)), rng.choice((0, 1)))


def parcels_hist(s, rng, n, test):
    cs = [test, test]
    while len(cs) < n:
        cs.append(parcels_rand(rng))
    rng.shuffle(cs)
    return cs


def parcels_p2(s, rng, test):
    # two cases whose answer differs between old and new rule, two Rule-1 cases, five uninsured
    def draw(f):   # never the test case itself (only the TEST slot may hold it)
        while (c := f()) == test:
            pass
        return c
    cs = ["TEST"] + [draw(lambda: (0, rng.choice((0, 1)), 1, rng.choice((0, 1)))) for _ in range(2)]
    cs += [draw(lambda: (1, rng.choice((0, 1)), rng.choice((0, 1)), rng.choice((0, 1)))) for _ in range(2)]
    cs += [draw(lambda: (0, rng.choice((0, 1)), 0, rng.choice((0, 1)))) for _ in range(5)]
    rng.shuffle(cs)
    stand = (test[0], test[1], 0, test[3])
    return cs, (stand if stand != test else (test[0], 1 - test[1], 0, test[3]))


def parcels_labels(s, c, rng):
    old, new = parcels_ans(s, c, False), parcels_ans(s, c, True)
    pool = [x for x in s["lab"] if x not in (old, new)]
    labs = [old] + ([new] if new != old else []) + rng.sample(pool, 3 - (1 + (new != old)))
    rng.shuffle(labs)
    return labs


PARCELS = dict(mk=parcels_set, rules=parcels_rules, ans=parcels_ans, case=parcels_case, probes=parcels_probes, hist=parcels_hist,
               p2=parcels_p2, labels=parcels_labels,
               update=lambda s: ("IMPORTANT UPDATE: From this point on, Rule 2 has been changed. New RULE 2: If Rule 1 does not apply, "
                                 f"a parcel that is insured and, at the same time, standard goes by {s['lab'][1]}. All other rules stay the same.\n\n"),
               review=lambda s: ("IMPORTANT UPDATE: Rule 2 has been reviewed and stays the same: If Rule 1 does not apply, a parcel that is "
                                 f"insured and, at the same time, express goes by {s['lab'][1]}. All other rules stay the same.\n\n"))


# ---------------- bonus: case = tuple of colours present (one token each) ----------------
def bonus_set(rng):
    col = rng.sample(COLOURS, 4)
    val = {c: rng.choice((1, 2)) for c in col}
    return dict(kind="bonus", cols=col, val=val, x=col[0], y=col[1])


def bonus_rules(s, new):
    lines = "\n".join(f"- {art(c)} {c} token is worth {s['val'][c]} point{'s' if s['val'][c] > 1 else ''}" for c in s["cols"])
    yy = f"no {s['y']} token" if new else f"{art(s['y'])} {s['y']} token"
    return ("RESEARCH TASK: TOKEN POINTS WITH A BONUS\n\nUse ONLY the following fictional scoring system. A case is a small pile of "
            f"coloured tokens, at most one of each colour.\n\nVALUES:\n{lines}\n\n"
            f"BONUS RULE: If a pile contains {art(s['x'])} {s['x']} token and, at the same time, {yy}, add 3 bonus points.\n\n"
            "The score of a case is the sum of the values of its tokens plus any bonus. Answer with the score as a single number.\n\n")


def bonus_ans(s, c, new):
    b = s["x"] in c and ((s["y"] not in c) if new else (s["y"] in c))
    return str(sum(s["val"][x] for x in c) + 3 * b)


def bonus_case(s, c):
    return ", ".join(f"1 {x}" for x in c)


def bonus_norm(s, cs):
    return tuple(x for x in s["cols"] if x in cs)


def bonus_rand(s, rng):
    return bonus_norm(s, rng.sample(s["cols"], rng.choice((1, 2, 3))))


def bonus_probes(s, rng):
    others = s["cols"][2:]
    t1 = bonus_norm(s, [s["x"], s["y"]])
    t2 = bonus_norm(s, [s["x"], s["y"], rng.choice(others)])
    s1 = bonus_norm(s, [s["y"], rng.choice(others)])
    s2 = bonus_norm(s, rng.sample(others, rng.choice((1, 2))))
    return {"T1": t1, "T2": t2, "S1": s1, "S2": s2}


def bonus_hist(s, rng, n, test):
    cs = [test, test]
    while len(cs) < n:
        cs.append(bonus_rand(s, rng))
    rng.shuffle(cs)
    return cs


def bonus_p2(s, rng, test):
    others = s["cols"][2:]
    def draw(f):   # never the test case itself (only the TEST slot may hold it)
        while (c := f()) == test:
            pass
        return c
    flips = [draw(lambda: bonus_norm(s, [s["x"]] + rng.sample([s["y"]] + others, rng.choice((0, 1, 2))))) for _ in range(2)]
    neutral = [draw(lambda: bonus_norm(s, rng.sample([s["y"]] + others, rng.choice((1, 2))))) for _ in range(7)]
    cs = ["TEST"] + flips + neutral
    rng.shuffle(cs)
    stand = bonus_norm(s, [c for c in test if c != s["x"]]) or (s["y"],)
    return cs, (stand if stand != test else draw(lambda: bonus_norm(s, rng.sample(others, 1))))


def bonus_labels(s, c, rng):
    old, new = bonus_ans(s, c, False), bonus_ans(s, c, True)
    pool = [str(d) for d in range(1, 10) if str(d) not in (old, new)]
    labs = [old] + ([new] if new != old else []) + rng.sample(pool, 3 - (1 + (new != old)))
    rng.shuffle(labs)
    return labs


BONUS = dict(mk=bonus_set, rules=bonus_rules, ans=bonus_ans, case=bonus_case, probes=bonus_probes, hist=bonus_hist,
             p2=bonus_p2, labels=bonus_labels,
             update=lambda s: (f"IMPORTANT UPDATE: From this point on, the bonus rule has been changed. New BONUS RULE: If a pile contains "
                               f"{art(s['x'])} {s['x']} token and, at the same time, no {s['y']} token, add 3 bonus points. All other values stay the same.\n\n"),
             review=lambda s: (f"IMPORTANT UPDATE: The bonus rule has been reviewed and stays the same: If a pile contains {art(s['x'])} {s['x']} token "
                               f"and, at the same time, {art(s['y'])} {s['y']} token, add 3 bonus points. All other values stay the same.\n\n"))

T = {"parcels": PARCELS, "bonus": BONUS}


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
                P = {"A60": (head + intro1 + rows(t, s, p1, False, 1) + "\n\n" + t["update"](s) + q, new)}
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
                                      changed=(old != new)))
    return items


if __name__ == "__main__":
    seed = int(sys.argv[1])
    items = build(seed)
    out = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "sets_new2.json")
    json.dump(dict(seed=seed, items=items), open(out, "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts ->", out, "| max chars", max(len(x["prompt"]) for x in items))
