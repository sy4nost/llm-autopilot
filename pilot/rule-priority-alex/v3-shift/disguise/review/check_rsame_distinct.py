"""Distinct R-same prompts (everything before 'Q:' removed) per template, test split only. python3 -I check_rsame_distinct.py <redefine-math.jsonl>"""
import json, sys, random
rows = [json.loads(l) for l in open(sys.argv[1])]
dev = set()
for t in range(9):
    dev |= set(random.Random(12345 + t).sample(range(t * 100, t * 100 + 100), 10))
tot, allp = 0, set()
for t in range(9):
    ps = ["Q:" + rows[i]["prompt"].split("Q:", 1)[1] for i in range(t * 100, t * 100 + 100) if i not in dev]
    tot += len(ps); allp |= set(ps)
    print(f"t{t}: test items {len(ps)}  distinct R-same prompts {len(set(ps))}  e.g. {ps[0]!r}")
print("total", tot, "distinct overall", len(allp))
