"""Exact match vs strategy (Alex's test): does 14B copy an identical old example, or keep the old strategy?
Same rules, update note and test format as v3 (make_shift.py). History length N = 20 or 60 in every condition.
Per set and N one base history is drawn; the conditions differ ONLY in the slots that held old-Rule-2 HOLD examples
(f1 open, f3 yes, f4 on; 15% of N), at the same positions:
  A exact       those slots = the test case itself (same four fields), labelled HOLD by the old rule
  B structural  those slots = old-Rule-2 HOLD cases that differ from the test case (the other f2 value)
  C none        those slots = HOLD via Rule 1 (f1 closed, never f3 yes + f4 on); no case uses old Rule 2
All other slots are identical across A/B/C (PASS, REVIEW, Rule-1 HOLD, never f3 yes + f4 on), so the label mix is
identical too. Probes: D1 (0,1,1,1), D3 (0,0,1,1) = old HOLD, new not HOLD; D2/D4 for contrast (history of D1/D3).
Arms: {A,B,C}_change_N (update note, key new rule), {A,B,C}_nochange_N (no note, key old rule), change_0, fresh.
python3 -I make_match.py <out.json>
"""
import json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_shift import (N_SETS, F1_NAMES, F2_NAMES, F3_NAMES, F4_NAMES, F1_VALS, F2_VALS, F3_VALS, F4_VALS, LABELS,
                        key, case_str, header, note, TAIL)

HIST = [20, 60]
PROBES = {"D1": (0, 1, 1, 1), "D2": (0, 1, 1, 0), "D3": (0, 0, 1, 1), "D4": (0, 0, 1, 0)}
GROUP = {"D1": 1, "D2": 1, "D3": 0, "D4": 0}  # f2 value of the old-Rule-2 test case whose history the probe uses


def base_history(rng, n_ex):
    n_h1, n_h2, n_r = round(.15 * n_ex), round(.15 * n_ex), round(.2 * n_ex)
    n_p = n_ex - n_h1 - n_h2 - n_r
    no_yo = lambda c: not (c[2] and c[3])
    nonhold = [(0, 1, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    rev = [(0, 0, a, b) for a in (0, 1) for b in (0, 1) if not (a and b)]
    h1 = [c for c in [(1, a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)] if no_yo(c)]
    cs = [rng.choice(nonhold) for _ in range(n_p)] + [rng.choice(h1) for _ in range(n_h1)]
    cs += ["SLOT"] * n_h2 + [rng.choice(rev) for _ in range(n_r)]
    rng.shuffle(cs)
    fill_c = [rng.choice(h1) for _ in range(n_h2)]  # condition C replacements
    return cs, fill_c, h1


def render(s, cs):
    lab = lambda k: s["lab"]["HPR".index(k)]
    rows = [f"Example {j:02d}: {case_str(s, c)} -> {lab(key(c, False))}" for j, c in enumerate(cs, 1)]
    return (f"Below are {len(cs)} solved examples. They show cases and their correct classifications.\n\n"
            + "\n".join(rows) + "\n\n")


def build(seed=20261020, keep=None):
    rng = random.Random(seed)
    items = []
    for i in range(N_SETS):
        s = dict(n=[rng.choice(F1_NAMES), rng.choice(F2_NAMES), rng.choice(F3_NAMES), rng.choice(F4_NAMES)],
                 v=[rng.choice(F1_VALS), rng.choice(F2_VALS), rng.choice(F3_VALS), rng.choice(F4_VALS)],
                 lab=rng.choice(LABELS), order=rng.sample(range(4), 4))
        hists = {}
        for n_ex in HIST:
            cs, fill_c, _ = base_history(rng, n_ex)
            for g in (0, 1):
                test = (0, g, 1, 1)
                other = (0, 1 - g, 1, 1)
                it_c = iter(fill_c)
                hists[("A", n_ex, g)] = [test if c == "SLOT" else c for c in cs]
                hists[("B", n_ex, g)] = [other if c == "SLOT" else c for c in cs]
                hists[("C", n_ex, g)] = [next(it_c) if c == "SLOT" else c for c in cs]
        if keep is not None:
            keep.append((s, hists))
        arms = [("fresh", None, 0, True, False), ("change_0", None, 0, True, True)]
        arms += [(f"{cond}_{a}_{n_ex}", cond, n_ex, a == "change", a == "change")
                 for cond in "ABC" for a in ("change", "nochange") for n_ex in HIST]
        for arm, cond, n_ex, key_new, with_note in arms:
            for pid, c in PROBES.items():
                hist = render(s, hists[(cond, n_ex, GROUP[pid])]) if cond else ""
                prompt = header(s, new=(arm == "fresh")) + hist + (note(s) if with_note else "") + \
                    TAIL.format(case=case_str(s, c))
                items.append(dict(level="S", set=i, arm=arm, prompt=prompt, labels=list(s["lab"]), fmt="single",
                                  max_new=20, probes=[dict(pos="P1", id=pid, key=s["lab"]["HPR".index(key(c, key_new))],
                                                           old=s["lab"]["HPR".index(key(c, False))])]))
    return items


if __name__ == "__main__":
    items = build()
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts; max chars", max(len(x["prompt"]) for x in items))
