"""v3 'shift' test for 14B: learn a two-condition rule from examples, then one line changes one word of Rule 2.
Does the model apply the new rule, or keep answering by the old one?

Old rule (= v2 level L2):  R1 HOLD if f1=CLOSED; R2 HOLD if f3=YES and f4=ON; R3 PASS if f2=BLUE; R4 REVIEW.
New rule (after the note): R2 becomes HOLD if f3=YES and f4=OFF (one word changed). Everything else stays.
Arms (one test case per prompt, as in v2 'single'):
  change_N    old rules + N solved examples (old rule) + UPDATE note + test case      key: new rule
  nochange_N  old rules + N solved examples + test case (no note)                      key: old rule
  fresh       new rule written directly in the rules, no examples, no note (gate)      key: new rule
  N in 0, 20, 60. Examples: ~50% PASS, ~15% HOLD via f1, ~15% HOLD via f3+f4, ~20% REVIEW (old rule).
Probes (f1 closed, f2 primary, f3 yes, f4 on):
  D1 (0,1,1,1) old HOLD  / new PASS      D2 (0,1,1,0) old PASS   / new HOLD
  D3 (0,0,1,1) old HOLD  / new REVIEW    D4 (0,0,1,0) old REVIEW / new HOLD
  A1 (1,1,0,0) HOLD both                 A2 (0,1,0,1) PASS both
python3 -I make_shift.py <out.json>
"""
import json, random, sys

N_SETS = 30
HIST = [0, 20, 60]
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
PROBES = [("D1", (0, 1, 1, 1)), ("D2", (0, 1, 1, 0)), ("D3", (0, 0, 1, 1)), ("D4", (0, 0, 1, 0)),
          ("A1", (1, 1, 0, 0)), ("A2", (0, 1, 0, 1))]


def key(c, new):
    if c[0] or (c[2] and (not c[3] if new else c[3])):
        return "H"
    return "P" if c[1] else "R"


def case_str(s, c):
    n, v = s["n"], s["v"]
    vals = [v[0][1] if c[0] else v[0][0], v[1][0] if c[1] else v[1][1],
            v[2][0] if c[2] else v[2][1], v[3][0] if c[3] else v[3][1]]
    return "; ".join(f"{n[k]}={vals[k]}" for k in s["order"])


def rules(s, new):
    n, v, (H, P, R) = s["n"], s["v"], s["lab"]
    f4 = v[3][1] if new else v[3][0]
    return (f"RULE 1 (highest priority): Label a case {H} if its {n[0]} is {v[0][1]}.\n"
            f"RULE 2: If Rule 1 does not apply, label the case {H} if its {n[2]} is {v[2][0]} and, at the same "
            f"time, its {n[3]} is {f4}.\n"
            f"RULE 3: If Rules 1 and 2 do not apply, label the case {P} if its {n[1]} is {v[1][0]}.\n"
            f"RULE 4: All remaining cases are {R}.\n")


def header(s, new):
    n, v = s["n"], s["v"]
    fields = ", ".join(f"{n[k]} ({'/'.join(v[k])})" for k in s["order"])
    return ("RESEARCH TASK: RULE-SWITCHING TEST\n\n"
            "Use ONLY the following fictional classification system. Apply its rules in the stated priority order, "
            "to every case independently.\n\n" + rules(s, new) + "\n"
            f"Every case has four fields: {fields}. The field values are the entire available evidence. "
            "There are no additional exceptions.\n\n")


def history(rng, s, n_ex):
    if n_ex == 0:
        return ""
    lab = lambda k: s["lab"]["HPR".index(k)]
    n_h1, n_h2, n_r = round(.15 * n_ex), round(.15 * n_ex), round(.2 * n_ex)
    n_p = n_ex - n_h1 - n_h2 - n_r
    nonhold = [(0, 1, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    rev = [(0, 0, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    cs = [rng.choice(nonhold) for _ in range(n_p)]
    cs += [(1, rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 1)) for _ in range(n_h1)]
    cs += [(0, rng.randint(0, 1), 1, 1) for _ in range(n_h2)]
    cs += [rng.choice(rev) for _ in range(n_r)]
    rng.shuffle(cs)
    rows = [f"Example {j:02d}: {case_str(s, c)} -> {lab(key(c, False))}" for j, c in enumerate(cs, 1)]
    return (f"Below are {n_ex} solved examples. They show cases and their correct classifications.\n\n"
            + "\n".join(rows) + "\n\n")


def note(s):
    n, v, H = s["n"], s["v"], s["lab"][0]
    return (f"IMPORTANT UPDATE (applies to the TEST case below): Rule 2 has been changed. New RULE 2: If Rule 1 does "
            f"not apply, label the case {H} if its {n[2]} is {v[2][0]} and, at the same time, its {n[3]} is "
            f"{v[3][1]}. All other rules stay the same.\n\n")


TAIL = "Classify the TEST case below. Do not explain. Return only its label.\n\nTEST CASE: {case}\n\nReturn only the label."


def build(seed=20261018):
    rng = random.Random(seed)
    items = []
    for i in range(N_SETS):
        s = dict(n=[rng.choice(F1_NAMES), rng.choice(F2_NAMES), rng.choice(F3_NAMES), rng.choice(F4_NAMES)],
                 v=[rng.choice(F1_VALS), rng.choice(F2_VALS), rng.choice(F3_VALS), rng.choice(F4_VALS)],
                 lab=rng.choice(LABELS), order=rng.sample(range(4), 4))
        hist = {n_ex: history(rng, s, n_ex) for n_ex in HIST}  # same history for change and nochange
        arms = [("fresh", 0, True, False)] + [(f"{a}_{n_ex}", n_ex, a == "change", a == "change")
                                               for a in ("change", "nochange") for n_ex in HIST]
        for arm, n_ex, key_new, with_note in arms:
            for pid, c in PROBES:
                base = header(s, new=(arm == "fresh")) + ("" if arm == "fresh" else hist[n_ex])
                prompt = base + (note(s) if with_note else "") + TAIL.format(case=case_str(s, c))
                k = s["lab"]["HPR".index(key(c, key_new))]
                old = s["lab"]["HPR".index(key(c, False))]
                items.append(dict(level="S", set=i, arm=arm, prompt=prompt, labels=list(s["lab"]), fmt="single",
                                  max_new=20, probes=[dict(pos="P1", id=pid, key=k, old=old)]))
    return items


if __name__ == "__main__":
    items = build()
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts; max chars", max(len(x["prompt"]) for x in items))
