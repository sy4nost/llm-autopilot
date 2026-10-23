"""Full run in the one-case-per-prompt format: every arm of sets.json, each of the 6 test cases as its own prompt.
The examples block stays exactly as in sets.json; only the instructions and the test part change.
python3 -I make_single.py sets.json sets_single.json
"""
import json, re, sys

FMT = "Do not explain. Return exactly six lines formatted P1: LABEL through P6: LABEL.\n\n"
out = []
for x in json.load(open(sys.argv[1])):
    p = x["prompt"]
    head = p.split("TEST CASES (unanswered):")[0]
    head = head.replace("classify the six TEST cases at the end. " + FMT, "classify the TEST case at the end. "
                        "Do not explain. Return only its label.\n\n")
    head = head.replace("Classify the six TEST cases below. " + FMT, "Classify the TEST case below. "
                        "Do not explain. Return only its label.\n\n")
    head = head.replace(FMT, "Do not explain. Return only the label of the TEST case.\n\n")  # control intro
    assert "six" not in head, head[-400:]
    for (j, case), pr in zip(re.findall(r"^P(\d): (.*)$", p, re.M), x["probes"]):
        out.append(dict(x, fmt="single", max_new=20, probes=[dict(pr, pos="P1")],
                        prompt=head + "TEST CASE: " + case + "\n\nReturn only the label."))
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(out), "prompts")
