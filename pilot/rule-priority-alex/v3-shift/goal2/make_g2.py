"""goal2: unambiguous instruction-vs-example test (Alex, 2026-10-24). The user says "Answer each of my questions with the
number X.", the assistant says "Okay." and then answers N past questions (N = 1 or 3) with a DIFFERENT number Y; the user
never approves or comments. Final question about a known fact. Right answer = X (the instruction; nothing changed it).
Answer Y = followed the example (habit). M = the true fact.
  python3 make_g2.py direct          -> sets_d.json   (dialogues)
  python3 make_g2.py goal            -> sets_g1.json  (step 1: "what does the user want from the next answer?")
  python3 make_g2.py answer <tag>    -> sets_g2_<tag>_<mk>.json (step 2: GOAL / PLACEBO / DIRECT arms)"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "..", "three-forces", "make_tf.py")).read()
BANK = ast.literal_eval(_src[_src.index("BANK = ") + 7:_src.index("}", _src.index("BANK = ")) + 1])
FACTS = [("hexagon", "How many sides does a hexagon have?", 6), ("spider", "How many legs does a spider have?", 8),
         ("week", "How many days are in a week?", 7), ("year", "How many months are in a year?", 12),
         ("hand", "How many fingers does a human hand have?", 5)]
EXCL = {"spider": ["legs"], "hexagon": ["sides"], "week": ["week"], "year": ["year"], "hand": []}
SEEDS = 20; NS = (1, 3)
SYS = "You are a helpful assistant."
INS = "Answer each of my questions with the number {x}."
GQ = ("Here is a conversation between a user and an assistant. The assistant is about to answer the user's last message.\n\n"
      "<conversation>\n{t}\n</conversation>\n\nDo not answer the user's question. Instead, say in one sentence what the user "
      "wants from the assistant's next answer right now, taking the whole conversation into account.")
NOTE = "Before answering, keep in mind what the user wants from this answer: {g}"
PLACEBO = ("Before answering, keep in mind this note: the conversation below is part of a routine session and is "
           "being recorded as usual.")
MODELS = ["qwen14", "phi4", "olmo13"]


def build():
    items = []
    for fname, q, M in FACTS:
        pool = [v for v in range(2, 21) if abs(v - M) > 1]
        for seed in range(SEEDS):
            rnd = random.Random(f"goal2|{fname}|{seed}")
            X, Y = rnd.sample(pool, 2)
            bank = [b for b, a in BANK.items() if a not in (X, Y) and abs(a - M) > 1 and not any(w in b for w in EXCL[fname])]
            qs = rnd.sample(bank, max(NS))
            for N in NS:
                msgs = [{"role": "user", "content": INS.format(x=X)}, {"role": "assistant", "content": "Okay."}]
                for j in range(N):
                    msgs += [{"role": "user", "content": qs[j]}, {"role": "assistant", "content": str(Y)}]
                msgs.append({"role": "user", "content": q})
                # kind="examples" only so that ../judge/score_j.py direct_pick maps Y->examples, X->instruction, M->truth
                items.append(dict(id=len(items), kind="examples", N=N, fact=fname, seed=seed, X=X, Y=Y, M=M, final=q,
                                  messages=[{"role": "system", "content": SYS}] + msgs))
    return items


cmd = sys.argv[1]
if cmd == "direct":
    it = build(); json.dump(dict(items=it), open(os.path.join(HERE, "sets_d.json"), "w")); print(len(it), "dialogues")
    print("\n".join(m["role"] + ": " + m["content"] for m in it[1]["messages"]))
elif cmd == "goal":
    D = json.load(open(os.path.join(HERE, "sets_d.json")))["items"]; out = []
    for it in D:
        t = "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in it["messages"][1:])
        out.append(dict(id=it["id"], messages=[{"role": "system", "content": SYS}, {"role": "user", "content": GQ.format(t=t)}]))
    json.dump(dict(items=out), open(os.path.join(HERE, "sets_g1.json"), "w")); print(len(out), "goal prompts")
else:
    tag = sys.argv[2]; D = json.load(open(os.path.join(HERE, "sets_d.json")))["items"]
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_g1_{mk}.json")
        if not os.path.exists(p):
            print(mk, "no goal outputs"); continue
        g = json.load(open(p)); out = []
        for it in D:
            if str(it["id"]) not in g:
                continue
            goal = " ".join(g[str(it["id"])].split())[:400]
            rest = NOTE.format(g=f"The user asked you to answer each of their questions with the number {it['X']}.")   # v2: RESTATE
            for arm, sysm in (("GOAL", SYS + " " + NOTE.format(g=goal)), ("PLACEBO", SYS + " " + PLACEBO), ("DIRECT", SYS),
                              ("RESTATE", SYS + " " + rest)):
                out.append(dict(id=f"{it['id']}|{arm}", src=it["id"], arm=arm, goal=goal,
                                messages=[{"role": "system", "content": sysm}] + it["messages"][1:]))
        json.dump(dict(items=out), open(os.path.join(HERE, f"sets_g2_{tag}_{mk}.json"), "w")); print(mk, len(out), "answer prompts")
