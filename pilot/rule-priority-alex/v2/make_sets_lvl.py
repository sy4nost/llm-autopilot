"""Rule-priority test v2: three difficulty levels, so every model can be measured on a rule it solves without examples.

Levels (same 4 fields everywhere, so prompts have the same shape and length):
  L1  one HOLD condition:      HOLD if f1 is CLOSED; else PASS if f2 is BLUE; else REVIEW. (f3, f4 irrelevant)
  L2  two HOLD rules, written separately: Rule 1 HOLD if f1 CLOSED; Rule 2 HOLD if f3 YES and f4 ON;
      Rule 3 PASS if BLUE; Rule 4 REVIEW.
  L3  Alex's original wording: HOLD if f1 CLOSED OR (f3 YES AND f4 ON); else PASS if BLUE; else REVIEW.
Arms:
  none    rules only (gate: does the model understand the rule?)
  exp     20 PASS examples (no HOLD ever shown)
  mixed   10 PASS + 5 HOLD (only via f1) + 5 REVIEW            (as in Alex's mixed prompt)
  mixed2  10 PASS + 3 HOLD via f1 + 2 HOLD via f3+f4 (non-BLUE tag) + 5 REVIEW (L2, L3 only: is the f3+f4 path dropped
          because it was never shown?)
  control 20 unrelated letter examples
Examples never contain f3=YES with f4=ON except the mixed2 HOLD rows.
30 sets per level, names/values/labels/orders randomised as in v1.  python3 -I make_sets_lvl.py <out.json>
"""
import json, random, sys

N_SETS = 30
F1_NAMES = ["seal", "latch", "gate", "hatch", "valve", "lid"]
F2_NAMES = ["tag", "badge", "stripe", "flag", "band", "mark"]
F3_NAMES = ["waiver", "permit", "token", "voucher", "pass", "ticket"]
F4_NAMES = ["echo", "pulse", "beacon", "signal", "relay", "chime"]
F1_VALS = [("OPEN", "CLOSED"), ("UNLOCKED", "LOCKED"), ("LOOSE", "SHUT"), ("FREE", "SEALED")]
F2_VALS = [("BLUE", "AMBER"), ("GREEN", "GREY"), ("SILVER", "BROWN"), ("WHITE", "VIOLET")]
F3_VALS = [("YES", "NO"), ("TRUE", "FALSE"), ("PRESENT", "ABSENT"), ("GIVEN", "MISSING")]
F4_VALS = [("ON", "OFF"), ("LIVE", "IDLE"), ("HIGH", "LOW"), ("ACTIVE", "QUIET")]
LABELS = [("HOLD", "PASS", "REVIEW"), ("STOP", "GO", "CHECK"), ("FREEZE", "CLEAR", "AUDIT"),
          ("BLOCK", "ALLOW", "INSPECT"), ("HALT", "PROCEED", "ESCALATE"), ("KEEP", "RELEASE", "QUERY")]
WORDS = ("MINT RAMP DUSK LAMP POND SILK TENT WAVE BARK RING FROG KITE SNOW PINE GLOW SAND MOON CAVE ROPE FIRE "
         "LEAF CORN DOVE HILL JADE KELP LIME MAST NEST OVAL PEAR QUIZ REEF SOCK TUSK VINE WOLF YARN ZINC BOLT").split()

# probe: (id, f1 closed, f2 primary, f3 yes, f4 on); key is computed from the level's rule
PROBES = {
    "L1": [("Q1", 1, 1, 0, 0), ("Q2", 0, 1, 0, 1), ("Q3", 0, 0, 0, 0), ("Q4", 1, 0, 1, 0),
           ("Q5", 0, 1, 1, 1), ("Q6", 1, 1, 1, 1)],   # Q1, Q6: CLOSED + BLUE -> HOLD (lure PASS); Q5 irrelevant fields
    "L2": [("P1", 1, 1, 0, 0), ("P2", 0, 1, 0, 1), ("P3", 0, 1, 1, 1), ("P4", 0, 0, 0, 0),
           ("P5", 0, 1, 1, 0), ("P6", 1, 0, 1, 0)],   # Alex's six probes
}
PROBES["L3"] = PROBES["L2"]
PRIMARY = {"L1": ["Q1", "Q6"], "L2": ["P1", "P3"], "L3": ["P1", "P3"]}


def key(level, c):
    if c[0] or (level != "L1" and c[2] and c[3]):
        return "H"
    return "P" if c[1] else "R"


def case_str(s, c):
    n, v = s["n"], s["v"]
    vals = [v[0][1] if c[0] else v[0][0], v[1][0] if c[1] else v[1][1],
            v[2][0] if c[2] else v[2][1], v[3][0] if c[3] else v[3][1]]
    return "; ".join(f"{n[k]}={vals[k]}" for k in s["order"])


def rules(level, s):
    n, v, (H, P, R) = s["n"], s["v"], s["lab"]
    if level == "L1":
        return (f"RULE 1 (highest priority): Label a case {H} if its {n[0]} is {v[0][1]}.\n"
                f"RULE 2: If Rule 1 does not apply, label the case {P} if its {n[1]} is {v[1][0]}.\n"
                f"RULE 3: All remaining cases are {R}.\n")
    if level == "L2":
        return (f"RULE 1 (highest priority): Label a case {H} if its {n[0]} is {v[0][1]}.\n"
                f"RULE 2: If Rule 1 does not apply, label the case {H} if its {n[2]} is {v[2][0]} and, at the same "
                f"time, its {n[3]} is {v[3][0]}.\n"
                f"RULE 3: If Rules 1 and 2 do not apply, label the case {P} if its {n[1]} is {v[1][0]}.\n"
                f"RULE 4: All remaining cases are {R}.\n")
    return (f"RULE 1 (highest priority): Label a case {H} if its {n[0]} is {v[0][1]}, OR if its {n[2]} is "
            f"{v[2][0]} AND its {n[3]} is {v[3][0]}.\n"
            f"RULE 2: If Rule 1 does not apply, label the case {P} if its {n[1]} is {v[1][0]}.\n"
            f"RULE 3: All remaining cases are {R}.\n")


def header(level, s):
    n, v = s["n"], s["v"]
    fields = ", ".join(f"{n[k]} ({'/'.join(v[k])})" for k in s["order"])
    return ("RESEARCH TASK: RULE-SWITCHING TEST\n\n"
            "Use ONLY the following fictional classification system. Apply its rules in the stated priority order, "
            "to every case independently.\n\n" + rules(level, s) + "\n"
            f"Every case has four fields: {fields}. The field values are the entire available evidence. "
            "There are no additional exceptions.\n\n")


FMT = "Do not explain. Return exactly six lines formatted P1: LABEL through P6: LABEL.\n\n"


def examples(rng, s, arm):
    if arm == "control":
        rows = [f"Example {j:02d}: word={w}; spare word={rng.choice(WORDS)}; requested position=2; answer={w[1]}"
                for j, w in enumerate(rng.sample(WORDS, 20), 1)]
        return ("For the 20 solved examples below, the task is to read the second letter of a four-letter word. "
                "These examples are unrelated to the classification system above. " + FMT + "\n".join(rows) + "\n\n")
    nonhold = [(0, 1, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    rev = [(0, 0, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    lab = lambda k: s["lab"]["HPR".index(k)]
    if arm == "exp":
        cs = [(rng.choice(nonhold), "P") for _ in range(20)]
    else:
        n_f1 = 5 if arm == "mixed" else 3
        cs = [(rng.choice(nonhold), "P") for _ in range(10)]
        cs += [((1, rng.randint(0, 1), 0, rng.randint(0, 1)), "H") for _ in range(n_f1)]
        cs += [((0, 0, 1, 1), "H") for _ in range(5 - n_f1)]  # other tag, so no example equals probe P3
        cs += [(rng.choice(rev), "R") for _ in range(5)]
        rng.shuffle(cs)
    rows = [f"Example {j:02d}: {case_str(s, c)} -> {lab(k)}" for j, (c, k) in enumerate(cs, 1)]
    return ("Below are 20 solved examples. They show cases and their correct classifications. After reading them, "
            "classify the six TEST cases at the end. " + FMT + "\n".join(rows) + "\n\n")


def build(seed=20261017):
    rng = random.Random(seed)
    items = []
    for level in ("L1", "L2", "L3"):
        for i in range(N_SETS):
            s = dict(n=[rng.choice(F1_NAMES), rng.choice(F2_NAMES), rng.choice(F3_NAMES), rng.choice(F4_NAMES)],
                     v=[rng.choice(F1_VALS), rng.choice(F2_VALS), rng.choice(F3_VALS), rng.choice(F4_VALS)],
                     lab=rng.choice(LABELS), order=rng.sample(range(4), 4))
            probes = rng.sample(PROBES[level], 6)
            ptxt = "\n".join(f"P{j}: {case_str(s, p[1:])}" for j, p in enumerate(probes, 1))
            arms = ["none", "exp", "mixed", "control"] + (["mixed2"] if level != "L1" else [])
            for arm in arms:
                body = "Classify the six TEST cases below. " + FMT if arm == "none" else examples(rng, s, arm)
                prompt = (header(level, s) + body + "TEST CASES (unanswered):\n" + ptxt
                          + "\n\nReturn exactly six lines, P1 to P6, without explanations.")
                items.append(dict(level=level, set=i, arm=arm, prompt=prompt, labels=list(s["lab"]),
                                  probes=[dict(pos=f"P{j}", id=p[0], key=s["lab"]["HPR".index(key(level, p[1:]))])
                                          for j, p in enumerate(probes, 1)]))
    return items


if __name__ == "__main__":
    items = build()
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts")
    for lv in ("L1", "L2", "L3"):
        for arm in ("none", "exp", "mixed", "mixed2", "control"):
            L = [len(x["prompt"]) for x in items if x["level"] == lv and x["arm"] == arm]
            if L:
                print(lv, arm.ljust(8), "chars mean", sum(L) // len(L))
