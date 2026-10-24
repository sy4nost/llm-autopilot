"""goal: does making the model state the user's current wish (the "goal") fix habit errors, and does
"quick answer != answer with goal" work as a lamp?  Uses the 300 direct dialogues of ../judge/sets_j.json.
  step 1:  python3 make_g.py goal    -> sets_g1.json  (transcript + "what does the user want from the next answer?")
  step 2:  python3 make_g.py answer <tag>  -> sets_g2_<tag>.json (the dialogue, with the model's own goal sentence as a
           system note; each model gets its own goal sentences, from out_<tag>_g1_<mk>.json)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
J = [it for it in json.load(open(os.path.join(HERE, "..", "judge", "sets_j.json")))["items"] if it["mode"] == "direct"]
SYS = "You are a helpful assistant."
GQ = ("Here is a conversation between a user and an assistant. The assistant is about to answer the user's last message.\n\n"
      "<conversation>\n{t}\n</conversation>\n\nDo not answer the user's question. Instead, say in one sentence what the user "
      "wants from the assistant's next answer right now, taking the whole conversation into account.")
NOTE = "Before answering, keep in mind what the user wants from this answer: {g}"
PLACEBO = ("Before answering, keep in mind this note: the conversation below is part of a routine session and is "
           "being recorded as usual.")
MODELS = ["qwen14", "phi4", "olmo13"]

if sys.argv[1] == "goal":
    out = []
    for it in J:
        msgs = it["messages"][1:]
        t = "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs)
        out.append(dict(id=it["id"], kind=it["kind"], messages=[{"role": "system", "content": SYS}, {"role": "user", "content": GQ.format(t=t)}]))
    json.dump(dict(items=out), open(os.path.join(HERE, "sets_g1.json"), "w")); print(len(out), "goal prompts"); print(out[0]["messages"][1]["content"])
else:
    tag = sys.argv[2]
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_g1_{mk}.json")
        if not os.path.exists(p):
            print(mk, "no goal outputs"); continue
        g = json.load(open(p)); out = []
        for it in J:
            if str(it["id"]) not in g:
                continue
            goal = " ".join(g[str(it["id"])].split())[:400]
            for arm, sysm in (("GOAL", SYS + " " + NOTE.format(g=goal)), ("PLACEBO", SYS + " " + PLACEBO), ("DIRECT", SYS)):
                out.append(dict(id=f"{it['id']}|{arm}", src=it["id"], arm=arm, kind=it["kind"], goal=goal,
                                messages=[{"role": "system", "content": sysm}] + it["messages"][1:]))   # v2: placebo + DIRECT re-run
        json.dump(dict(items=out), open(os.path.join(HERE, f"sets_g2_{tag}_{mk}.json"), "w")); print(mk, len(out), "answer prompts")
