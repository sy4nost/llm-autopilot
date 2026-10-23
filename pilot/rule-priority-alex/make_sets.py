"""Generate 30 problem sets for Alex's HOLD/PASS/REVIEW rule-priority test.

Set 0 is Alex's original wording (field and label names, probe order). Sets 1..29 keep the same logic but
change field names, value words, label words, field order, example order and probe order.

Arms per set:
  exp     20 examples, all PASS (as in Alex's experimental_prompt)
  mixed   10 PASS + 5 HOLD (all via the first field) + 5 REVIEW (as in Alex's mixed_prompt)
  control 20 unrelated letter examples, padded so the prompt is about as long as exp
  none    rules + probes, no examples (baseline)

Logical probes (same 6 as Alex's key; `kind` is what the first-label check looks at):
  P1 closed, blue, no, off      -> H   (lure: blue -> P)
  P2 open, blue, no, on         -> P
  P3 open, blue, yes, on        -> H   (lure: blue -> P; this HOLD path never appears in any example)
  P4 open, amber, no, off       -> R
  P5 open, blue, yes, off       -> P   (lure: waiver -> H)
  P6 closed, amber, yes, off    -> H
  python3 -I make_sets.py <out.json> <dir with Alex's original prompts>
"""
import json, random, sys

N_SETS = 30
PROBES = [  # (id, f1 closed?, f2 primary?, f3 yes?, f4 on?, key)
    ("P1", 1, 1, 0, 0, "H"), ("P2", 0, 1, 0, 1, "P"), ("P3", 0, 1, 1, 1, "H"),
    ("P4", 0, 0, 0, 0, "R"), ("P5", 0, 1, 1, 0, "P"), ("P6", 1, 0, 1, 0, "H")]

F1_NAMES = ["seal", "latch", "gate", "hatch", "valve", "lid"]
F2_NAMES = ["tag", "badge", "stripe", "flag", "band", "mark"]
F3_NAMES = ["waiver", "permit", "token", "voucher", "pass", "ticket"]
F4_NAMES = ["echo", "pulse", "beacon", "signal", "relay", "chime"]
F1_VALS = [("OPEN", "CLOSED"), ("UNLOCKED", "LOCKED"), ("LOOSE", "SHUT"), ("FREE", "SEALED")]  # (default, trigger)
F2_VALS = [("BLUE", "AMBER"), ("GREEN", "GREY"), ("SILVER", "BROWN"), ("WHITE", "VIOLET")]   # (primary, other)
F3_VALS = [("YES", "NO"), ("TRUE", "FALSE"), ("PRESENT", "ABSENT"), ("GIVEN", "MISSING")]  # (yes, no)
F4_VALS = [("ON", "OFF"), ("LIVE", "IDLE"), ("HIGH", "LOW"), ("ACTIVE", "QUIET")]          # (on, off)
LABELS = [("HOLD", "PASS", "REVIEW"), ("STOP", "GO", "CHECK"), ("FREEZE", "CLEAR", "AUDIT"),
          ("BLOCK", "ALLOW", "INSPECT"), ("HALT", "PROCEED", "ESCALATE"), ("KEEP", "RELEASE", "QUERY")]
WORDS = ("MINT RAMP DUSK LAMP POND SILK TENT WAVE BARK RING FROG KITE SNOW PINE GLOW SAND MOON CAVE ROPE FIRE "
         "LEAF CORN DOVE HILL JADE KELP LIME MAST NEST OVAL PEAR QUIZ REEF SOCK TUSK VINE WOLF YARN ZINC BOLT").split()


def make_scheme(rng, i):
    if i == 0:
        f = dict(n=["seal", "tag", "waiver", "echo"], v=[F1_VALS[0], F2_VALS[0], F3_VALS[0], F4_VALS[0]],
                 lab=LABELS[0], order=[0, 1, 2, 3])
    else:
        f = dict(n=[rng.choice(F1_NAMES), rng.choice(F2_NAMES), rng.choice(F3_NAMES), rng.choice(F4_NAMES)],
                 v=[rng.choice(F1_VALS), rng.choice(F2_VALS), rng.choice(F3_VALS), rng.choice(F4_VALS)],
                 lab=rng.choice(LABELS), order=rng.sample(range(4), 4))
    return f


def case_str(s, c):  # c = (f1 closed, f2 primary, f3 yes, f4 on)
    n, v = s["n"], s["v"]
    vals = [v[0][1] if c[0] else v[0][0], v[1][0] if c[1] else v[1][1],
            v[2][0] if c[2] else v[2][1], v[3][0] if c[3] else v[3][1]]
    return "; ".join(f"{n[k]}={vals[k]}" for k in s["order"])


def lab(s, k):
    return s["lab"]["HPR".index(k)]


def header(s):
    n, v, (H, P, R) = s["n"], s["v"], s["lab"]
    fields = ", ".join(f"{n[k]} ({'/'.join(v[k])})" for k in s["order"])
    return (
        "RESEARCH TASK: RULE-SWITCHING TEST\n\n"
        "Use ONLY the following fictional classification system. Apply its rules in the stated priority order, "
        "to every case independently.\n\n"
        f"RULE 1 (highest priority): Label a case {H} if its {n[0]} is {v[0][1]}, OR if its {n[2]} is {v[2][0]} "
        f"AND its {n[3]} is {v[3][0]}.\n"
        f"RULE 2: If Rule 1 does not apply, label the case {P} if its {n[1]} is {v[1][0]}.\n"
        f"RULE 3: All remaining cases are {R}.\n\n"
        f"Every case has four fields: {fields}. The field values are the entire available evidence. "
        "There are no additional exceptions.\n\n")


def examples(rng, s, arm):
    if arm == "none":
        return ""
    if arm == "control":
        intro = ("For the 20 solved examples below, the task is to read the second letter of a four-letter word. "
                 "These examples are unrelated to the classification system above.\n\n")
        rows = []
        for j, w in enumerate(rng.sample(WORDS, 20), 1):
            w2 = rng.choice(WORDS)
            rows.append(f"Example {j:02d}: word={w}; spare word={w2}; requested position=2; answer={w[1]}")
        return intro + "\n".join(rows) + "\n\n"
    nonhold = [c for c in [(0, 1, a, b) for a in (0, 1) for b in (0, 1)] if not (c[2] and c[3])]
    if arm == "exp":
        cs = [(rng.choice(nonhold), "P") for _ in range(20)]
    else:  # mixed
        cs = [(rng.choice(nonhold), "P") for _ in range(10)]
        cs += [((1, rng.randint(0, 1), 0, rng.randint(0, 1)), "H") for _ in range(5)]
        cs += [((0, 0, a, b), "R") for a, b in [rng.choice([(0, 0), (0, 1), (1, 0)]) for _ in range(5)]]
        rng.shuffle(cs)
    intro = ("Below are 20 solved examples. They show cases and their correct classifications. After reading them, "
             "classify the six TEST cases at the end.\n\n")
    rows = [f"Example {j:02d}: {case_str(s, c)} -> {lab(s, k)}" for j, (c, k) in enumerate(cs, 1)]
    return intro + "\n".join(rows) + "\n\n"


def build(seed=20261016):
    rng = random.Random(seed)
    items = []
    for i in range(N_SETS):
        s = make_scheme(rng, i)
        probes = PROBES[:] if i == 0 else rng.sample(PROBES, 6)
        ptxt = "\n".join(f"P{j}: {case_str(s, p[1:5])}" for j, p in enumerate(probes, 1))
        for arm in ("exp", "mixed", "control", "none"):
            ex = examples(rng, s, arm)
            instr = ("Do not explain. Return exactly six lines formatted P1: LABEL through P6: LABEL.\n\n"
                     if arm == "none" else "")
            prompt = (header(s) + (ex if arm != "none" else "Classify the six TEST cases below. ") + instr
                      + "TEST CASES (unanswered):\n" + ptxt
                      + "\n\nReturn exactly six lines, P1 to P6, without explanations.")
            if arm != "none":  # same sentence as Alex's prompts, placed before the examples
                prompt = prompt.replace("classify the six TEST cases at the end.\n\n",
                                        "classify the six TEST cases at the end. Do not explain. Return exactly "
                                        "six lines formatted P1: LABEL through P6: LABEL.\n\n")
            items.append(dict(set=i, arm=arm, prompt=prompt, labels=list(s["lab"]),
                              probes=[dict(pos=f"P{j}", id=p[0], key=lab(s, p[5]), key_hpr=p[5])
                                      for j, p in enumerate(probes, 1)]))
    return items


if __name__ == "__main__":
    items = build()
    # set 0: use Alex's original prompts word for word (same rules, fields and probe order)
    orig = dict(exp="experimental_prompt.txt", mixed="mixed_prompt.txt", control="control_prompt.txt")
    for x in items:
        if x["set"] == 0 and x["arm"] in orig:
            x["prompt"] = open(sys.argv[2] + "/" + orig[x["arm"]]).read().strip()
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for arm in ("exp", "mixed", "control", "none"):
        L = [len(x["prompt"]) for x in items if x["arm"] == arm]
        print(arm, "chars min/mean/max", min(L), sum(L) // len(L), max(L))
