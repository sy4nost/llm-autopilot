"""'Solve first, then consult experience' (Alex's idea) on the exact-match sets (make_match.py, same 30 sets).
Pass 1 (draft): rules (+ update note) + test case, NO examples. Greedy answer = the draft.
Pass 2 (check): rules + the same history as the A/B/C condition (+ note) + test case + the model's own draft, and
  "look at the examples and decide whether your past experience should change this answer". Neutral wording:
  it does not say whether the rules or the examples win.
Arms: tp_{A,B,C}_{change,nochange}_{20,60}; probes D1-D4. Pass-2 prompts hold '{draft}', filled in on the GPU.
Compare with the one-pass arms {A,B,C}_{change,nochange}_N in match/results.
python3 make_twopass.py <out.json>
"""
import json, sys
from make_match import build, render, PROBES, GROUP, HIST
from make_shift import key, case_str, header, note

CHECK = ("Your first answer to this TEST case, made using only the rules and without looking at the solved examples, "
         "was: {draft}\n\nNow look at the solved examples above and decide whether your past experience should change "
         "this answer. Check carefully. Return only the final label.")
P1_TAIL = "Classify the TEST case below. Do not explain. Return only its label.\n\nTEST CASE: {case}\n\nReturn only the label."

keep = []
build(keep=keep)
items = []
for i, (s, hists) in enumerate(keep):
    for change in (True, False):
        for pid, c in PROBES.items():
            p1 = header(s, new=False) + (note(s) if change else "") + P1_TAIL.format(case=case_str(s, c))
            k = s["lab"]["HPR".index(key(c, change))]
            old = s["lab"]["HPR".index(key(c, False))]
            for cond in "ABC":
                for n_ex in HIST:
                    p2 = (header(s, new=False) + render(s, hists[(cond, n_ex, GROUP[pid])]) + (note(s) if change else "")
                          + f"Classify the TEST case below.\n\nTEST CASE: {case_str(s, c)}\n\n" + CHECK)
                    items.append(dict(level="S", set=i, arm=f"tp_{cond}_{'change' if change else 'nochange'}_{n_ex}",
                                      prompt1=p1, prompt=p2, labels=list(s["lab"]), fmt="single", max_new=20,
                                      probes=[dict(pos="P1", id=pid, key=k, old=old)]))
if __name__ == "__main__":
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "pass-2 prompts;", len({x["prompt1"] for x in items}), "distinct pass-1 prompts")
