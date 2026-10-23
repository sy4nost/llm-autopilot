"""Gate-format check: take the `none` arm of sets.json and build two easier answer formats.
  reason  same prompt, but the model may reason before the six final lines (max 700 new tokens)
  single  one test case per prompt, answer one label (max 20 new tokens)
python3 -I make_gate.py sets.json sets_gate.json
"""
import json, re, sys

FMT = "Do not explain. Return exactly six lines formatted P1: LABEL through P6: LABEL.\n\n"
REASON = ("For each test case, check the rules in priority order and say briefly which rule applies. "
          "Then finish with exactly six lines formatted P1: LABEL through P6: LABEL.\n\n")
out = []
for x in json.load(open(sys.argv[1])):
    if x["arm"] != "none":
        continue
    p = x["prompt"].replace(FMT, REASON).replace("\n\nReturn exactly six lines, P1 to P6, without explanations.",
                                                  "\n\nEnd with the six lines P1 to P6.")
    out.append(dict(x, fmt="reason", prompt=p, max_new=700))
    head, cases = x["prompt"].split("Classify the six TEST cases below. ")[0], re.findall(r"^P(\d): (.*)$", x["prompt"], re.M)
    for (j, case), pr in zip(cases, x["probes"]):
        q = (head + "Classify the TEST case below. Do not explain. Return only its label.\n\n"
             "TEST CASE: " + case + "\n\nReturn only the label.")
        out.append(dict(x, fmt="single", prompt=q, max_new=20, probes=[dict(pr, pos="P1")]))
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(out), "prompts")
