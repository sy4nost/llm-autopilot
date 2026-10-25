"""pull: how hard does each force (instruction / examples / truth) pull, measured from outside?
Uses the 300 'direct' dialogues of ../judge/sets_j.json. For each dialogue three variants:
  FULL  - the dialogue as is
  NOEX  - the 3 past Q/A turns removed (the final user message kept unchanged)
  NOINS - the latest number instruction removed (instruction kind: the correction line; other kinds: first turn + "Okay.")
For each variant the model's probability of each of the three candidate numbers as the full answer is read
(instruction number, shown number, true M). python3 make_p.py -> sets_p.json"""
import json, os
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "judge", "sets_j.json")))["items"]
SENT = {"hexagon": "A hexagon has {} sides.", "spider": "A spider has {} legs.", "week": "There are {} days in a week.",
        "year": "There are {} months in a year.", "hand": "A human hand has {} fingers."}   # v2 (review R1): sentence form
out = []
for it in J:
    if it["mode"] != "direct":
        continue
    sysm, msgs = it["messages"][0], it["messages"][1:]
    k = it["kind"]
    cand = {"instruction": it["X2"] if k == "instruction" else it["X"],
            "examples": it["X"] if k == "instruction" else it["Y"], "truth": it["M"]}
    assert len(set(cand.values())) == 3
    head, ex, last = msgs[:2], msgs[2:-1], msgs[-1]
    assert len(ex) == 6 and head[1]["content"] == "Okay."
    noex = head + [last]
    if k == "instruction":
        assert "\n" in last["content"]
        noins = head + ex + [dict(last, content=last["content"].split("\n", 1)[1])]
    else:
        noins = ex + [last]
    for var, m in (("FULL", msgs), ("NOEX", noex), ("NOINS", noins)):
        out.append(dict(id=f"{it['id']}|{var}", src_id=it["id"], var=var, kind=k, fact=it["fact"], seed=it["seed"],
                        cand=cand, sent={src: SENT[it["fact"]].format(v) for src, v in cand.items()}, messages=[sysm] + m))
json.dump(dict(items=out), open("sets_p.json", "w"))
print(len(out), "prompts")
for o in out[:3] + out[3 * 100:3 * 100 + 3]:
    print("-----", o["id"], o["kind"], o["cand"]); print("\n".join(m["role"] + ": " + m["content"] for m in o["messages"][1:]))
