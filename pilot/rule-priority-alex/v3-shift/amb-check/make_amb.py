"""amb-check (Alex 2026-10-24 03:36Z): are parcels/bonus errors automatism or misreading? Same conftrap-lamp items (seed 57),
one change per arm. python3 -I make_amb.py -> sets_amb.json
  BASE    the original prompt
  REMIND  + before "TEST CASE:": "Apply the rules as they are NOW, after the IMPORTANT UPDATE."
  REPEAT  + before "TEST CASE:": the new-rule sentence copied again ("Remember: New RULE 2: ...")
  NOHIST  the 60 solved examples removed (audit baseline: the rules alone)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = json.load(open(os.path.join(HERE, "..", "conftrap-lamp", "sets_cl.json")))["items"]
items = []
for x in src:
    if x["family"] not in ("parcels", "bonus"):
        continue
    p = x["prompt"]; t = p.rindex("TEST CASE:")
    a = p.index("New "); b = p.index("same.", a) + len("same.")
    rule = p[a:p.index(".", a) + 1]   # the new-rule sentence itself
    e0 = p.index("Below are 60 solved examples") if "Below are 60 solved examples" in p else p.index("Example 01:")
    e1 = p.index("IMPORTANT UPDATE")
    arms = {"BASE": p,
            "REMIND": p[:t] + "Apply the rules as they are NOW, after the IMPORTANT UPDATE.\n\n" + p[t:],
            "REPEAT": p[:t] + "Remember: " + rule + "\n\n" + p[t:],
            "NOHIST": p[:e0] + p[e1:]}
    for k, q in arms.items():
        items.append(dict(id=len(items), src=x["id"], family=x["family"], pop=x["pop"], arm=k, prompt=q, cands=x["cands"], key=x["key"], habit=x["habit"]))
json.dump(dict(items=items), open(os.path.join(HERE, "sets_amb.json"), "w"), ensure_ascii=False)
ex = [i for i in items if i["src"] == items[0]["src"]]
for i in ex:
    print("=====", i["arm"]); print(i["prompt"][:200].replace("\n", "|")); print("...", i["prompt"][-450:].replace("\n", "|"))
print(len(items))
