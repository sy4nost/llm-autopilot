"""Corrected framing (Alex: the task pushed the model toward experience too hard). Same 30 sets and histories as
make_match.py, N=60 only. Two changes:
  1. examples are introduced as past cases, not as correct classifications:
     "Below are 60 past cases and the labels they were given at the time."
  2. the two-pass check no longer asks whether experience should change the answer; it says keep or change,
     whichever is correct under the rules that apply to the TEST case.
Arms: nt1_{A,B,C}_{change,nochange}_60 = one pass, neutral header;
      nt2_{A,B,C}_{change,nochange}_60 = draft without examples, then neutral check with the history.
python3 make_neutral.py <out.json>
"""
import json, sys
from make_match import build, render, PROBES, GROUP
from make_shift import key, case_str, header, note

OLD_HEAD = "Below are 60 solved examples. They show cases and their correct classifications."
NEW_HEAD = "Below are 60 past cases and the labels they were given at the time."
CHECK = ("Your first answer to this TEST case, made using only the rules and without looking at the past cases, "
         "was: {draft}\n\nNow look at the past cases above. Keep your first answer or change it, whichever is correct "
         "under the rules that apply to the TEST case. Return only the final label.")
TAIL = "Classify the TEST case below. Do not explain. Return only its label.\n\nTEST CASE: {case}\n\nReturn only the label."

keep = []
build(keep=keep)
items = []
for i, (s, hists) in enumerate(keep):
    for change in (True, False):
        ch = "change" if change else "nochange"
        for pid, c in PROBES.items():
            p1 = header(s, new=False) + (note(s) if change else "") + TAIL.format(case=case_str(s, c))
            pr = [dict(pos="P1", id=pid, key=s["lab"]["HPR".index(key(c, change))],
                       old=s["lab"]["HPR".index(key(c, False))])]
            for cond in "ABC":
                hist = render(s, hists[(cond, 60, GROUP[pid])])
                assert hist.startswith(OLD_HEAD)
                hist = NEW_HEAD + hist[len(OLD_HEAD):]
                base = header(s, new=False) + hist + (note(s) if change else "")
                one = base + TAIL.format(case=case_str(s, c))
                two = base + f"Classify the TEST case below.\n\nTEST CASE: {case_str(s, c)}\n\n" + CHECK
                for arm, p in ((f"nt1_{cond}_{ch}_60", one), (f"nt2_{cond}_{ch}_60", two)):
                    items.append(dict(level="S", set=i, arm=arm, prompt1=p1, prompt=p, labels=list(s["lab"]),
                                      fmt="single", max_new=40, probes=pr))
if __name__ == "__main__":
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts")
