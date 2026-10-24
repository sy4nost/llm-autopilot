"""cicm-goal: the "goal -> conflict -> slow re-check" scheme on Berkeley CICM (stale binding), short version.
Data: CICM primary set, Berkeley's own calibration split (240 dialogues, 40 per factorial cell), read from
../cicm-lamp/data (CC BY-NC, never pushed). The dialogue's last user message is the query ("what is the latest value…").
  python3 make_cg.py base          -> sets_g1.json (goal step) + sets_th.json (THINK arm, all items)
  python3 make_cg.py answer <tag>  -> sets_g2_<tag>_<mk>.json (GOAL / PLACEBO / DIRECT arms)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "cicm-lamp")
sys.path.insert(0, CL); import split_cc
ROWS = [json.loads(l) for l in open(os.path.join(CL, "data", "cicm_natural_factorial_otherdist_l0.jsonl"))]
CAL, _ = split_cc.split(ROWS)
GQ = ("Here is a conversation between a user and an assistant. The assistant is about to answer the user's last message.\n\n"
      "<conversation>\n{t}\n</conversation>\n\nDo not answer the user's question. Instead, say in one sentence what the user "
      "wants from the assistant's next answer right now, taking the whole conversation into account, and name the specific "
      "value the user expects the answer to contain.")   # v1.1 after smoke: without this, goals named no value (Qwen 92%)
NOTE = "Before answering, keep in mind what the user wants from this answer: {g}"
PLACEBO = ("Before answering, keep in mind this note: the conversation below is part of a routine session and is "
           "being recorded as usual.")
THINK = ("\n\nBefore answering, think step by step: go through the conversation in order and track every change to this "
         "preference. Then give the final value on the last line as \"Answer: <value>\".")
MODELS = ["qwen14", "phi4", "olmo13"]


def transcript(msgs):
    return "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs if m["role"] != "system")


cmd = sys.argv[1] if __name__ == "__main__" else None
if cmd == "base":
    g1, th = [], []
    for r in CAL:
        sysm = r["messages"][0]; assert sysm["role"] == "system" and r["messages"][-1]["role"] == "user"
        g1.append(dict(id=r["id"], messages=[{"role": "system", "content": "You are a helpful assistant."},
                                             {"role": "user", "content": GQ.format(t=transcript(r["messages"]))}]))
        last = dict(r["messages"][-1]); last["content"] = last["content"] + THINK
        th.append(dict(id=r["id"], messages=r["messages"][:-1] + [last]))
    json.dump(dict(items=g1), open(os.path.join(HERE, "sets_g1.json"), "w"))
    json.dump(dict(items=th), open(os.path.join(HERE, "sets_th.json"), "w"))
    print(len(g1), "dialogues"); print(th[0]["messages"][-1]["content"])
elif cmd == "answer":
    tag = sys.argv[2]
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_g1_{mk}.json")
        if not os.path.exists(p):
            print(mk, "no goal outputs"); continue
        g = json.load(open(p)); out = []
        for r in CAL:
            if r["id"] not in g:
                continue
            goal = " ".join(g[r["id"]].split())[:400]; s0 = r["messages"][0]["content"]
            for arm, sysm in (("GOAL", s0 + " " + NOTE.format(g=goal)), ("PLACEBO", s0 + " " + PLACEBO), ("DIRECT", s0)):
                out.append(dict(id=f"{r['id']}|{arm}", src=r["id"], arm=arm, goal=goal,
                                messages=[{"role": "system", "content": sysm}] + r["messages"][1:]))
        json.dump(dict(items=out), open(os.path.join(HERE, f"sets_g2_{tag}_{mk}.json"), "w")); print(mk, len(out), "answer prompts")
