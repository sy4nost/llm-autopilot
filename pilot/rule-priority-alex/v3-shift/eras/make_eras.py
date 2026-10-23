"""Two-era test for Alex's formula 'vote strength = case similarity x sameness of conditions'.
Same sets, rules, labels and PERIOD-1 history as match/ condition A (60 old-rule examples that include exact copies of
the D1/D3 test case). Held-out sets 15-29 only (the copy heads were picked on sets 0-14).
Arms (probes D1-D4 each):
  A60       exactly the match/ A_change_60 prompt (60 old examples, update note, test). For comparison with blockex2.
  E_copy    PERIOD 1 (same 60) + update note + PERIOD 2: 10 examples solved under the NEW rule, one of them the test case.
  E_nocopy  identical PERIOD 2 except the test-case slot holds a neutral stand-in (same f1/f2, f3 absent).
  N_same    no update: PERIOD 2 = 10 more examples under the OLD rule (incl. the test case); key = old rule. Control.
python3 -I make_eras.py -> sets_eras.json
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "match"))
from make_shift import key, case_str, header, TAIL  # noqa: E402
import make_match  # noqa: E402

PROBES = make_match.PROBES
GROUP = make_match.GROUP


def rows(s, cs, new, start):
    lab = lambda k: s["lab"]["HPR".index(k)]
    return "\n".join(f"Example {j:02d}: {case_str(s, c)} -> {lab(key(c, new))}" for j, c in enumerate(cs, start))


def update(s):
    n, v, H = s["n"], s["v"], s["lab"][0]
    return (f"IMPORTANT UPDATE: From this point on, Rule 2 has been changed. New RULE 2: If Rule 1 does not apply, label "
            f"the case {H} if its {n[2]} is {v[2][0]} and, at the same time, its {n[3]} is {v[3][1]}. "
            f"All other rules stay the same.\n\n")


def period2(rng, test):
    """10 cases drawn once per (set, probe): slot 'TEST' (the test case itself), two cases with f3 present and the other
    f4 value (Rule 2 status flips between rules), 2 Rule-1 cases, 5 with f3 absent. Returns the list with the slot
    marker and the neutral stand-in used for the no-copy arm (same f1/f2, f3 absent: no Rule 2 information)."""
    cs = ["TEST"] + [(0, rng.choice((0, 1)), 1, 1 - test[3]) for _ in range(2)]
    cs += [(1, rng.choice((0, 1)), rng.choice((0, 1)), rng.choice((0, 1))) for _ in range(2)]
    cs += [(0, rng.choice((0, 1)), 0, rng.choice((0, 1))) for _ in range(5)]
    rng.shuffle(cs)
    return cs, (test[0], test[1], 0, test[3])


def build():
    keep = []
    match_items = make_match.build(keep=keep)
    a60 = {(x["set"], x["probes"][0]["id"]): x for x in match_items if x["arm"] == "A_change_60"}
    rng = random.Random(20261017)
    items = []
    for i, (s, hists) in enumerate(keep):
        if i < 15:
            continue
        for pid, c in PROBES.items():
            p1 = hists[("A", 60, GROUP[pid])]
            items.append(dict(a60[(i, pid)], arm="A60"))
            slots, stand = period2(rng, c)
            for arm, copy, changed in (("E_copy", True, True), ("E_nocopy", False, True), ("N_same", True, False)):
                p2 = [(c if copy else stand) if x == "TEST" else x for x in slots]
                prompt = (header(s, new=False)
                          + f"Below are 60 solved examples from PERIOD 1. They show cases and their correct classifications at that time.\n\n"
                          + rows(s, p1, False, 1) + "\n\n"
                          + (update(s) if changed else "")
                          + f"Below are 10 solved examples from PERIOD 2"
                          + (" (after the update)" if changed else "") + ".\n\n"
                          + rows(s, p2, changed, 61) + "\n\n"
                          + TAIL.format(case=case_str(s, c)))
                items.append(dict(level="S", set=i, arm=arm, prompt=prompt, labels=list(s["lab"]), fmt="single", max_new=20,
                                  p1=p1, p2=p2, case=list(c),
                                  probes=[dict(pos="P1", id=pid, key=s["lab"]["HPR".index(key(c, changed))],
                                               old=s["lab"]["HPR".index(key(c, False))])]))
    return items


if __name__ == "__main__":
    items = build()
    json.dump(items, open(os.path.join(HERE, "sets_eras.json"), "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts; arms", sorted({x["arm"] for x in items}), "max chars", max(len(x["prompt"]) for x in items))
