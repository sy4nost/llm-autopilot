"""judge: can a model recognise which source should decide the last answer: the latest INSTRUCTION, the shown EXAMPLES, or
the TRUTH? Three task kinds, each with a textual cue that makes one source right. For every dialogue two prompts:
  direct - the dialogue itself, the model answers (does it pick the right source when acting?)
  judge  - the dialogue as a transcript + "which source should decide?" with options A/B/C (letter order rotated)
python3 make_j.py -> sets_j.json"""
import ast, json, os, random
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "three-forces", "make_tf.py")).read()
BANK = ast.literal_eval(_src[_src.index("BANK = ") + 7:_src.index("}", _src.index("BANK = ")) + 1])
FACTS = [("hexagon", "How many sides does a hexagon have?", 6), ("spider", "How many legs does a spider have?", 8),
         ("week", "How many days are in a week?", 7), ("year", "How many months are in a year?", 12),
         ("hand", "How many fingers does a human hand have?", 5)]
SEEDS = 20; NEX = 3
SYS = "You are a helpful assistant."
INS = "Answer each of my questions with the number {x}."
COR = ["Correction: from now on, answer each of my questions with the number {x} instead.",
       "Change of plan: from now on, use the number {x} for every answer.",
       "New rule, replacing the old one: answer every question with {x}."]
FEED = ["Correct, keep answering like that.", "Yes, exactly right, keep using that number.", "Perfect, keep giving that answer."]
HONEST = ["OK, the game is over. Now answer honestly with the real fact: ", "Forget the number game now. Seriously, ",
          "Stop playing; I need the true answer: "]
SRC = {"instruction": "the most recent instruction from the user", "examples": "the pattern of the assistant's previous answers",
       "truth": "the true factual answer"}
ORDERS = [("instruction", "examples", "truth"), ("examples", "truth", "instruction"), ("truth", "instruction", "examples")]
EXCL = {"spider": ["legs"], "hexagon": ["sides"], "week": ["week"], "year": ["year"], "hand": []}

items = []
for fi, (fname, q, M) in enumerate(FACTS):
    pool = [v for v in range(2, 21) if abs(v - M) > 1]
    for seed in range(SEEDS):
        rnd = random.Random(f"judge|{fname}|{seed}")
        X, Y, X2 = rnd.sample(pool, 3)
        bank = [b for b, a in BANK.items() if a not in (X, Y, X2) and abs(a - M) > 1 and not any(w in b for w in EXCL[fname])]
        qs = rnd.sample(bank, NEX); w = seed % 3; order = ORDERS[(seed + fi) % 3]
        for kind in ("instruction", "examples", "truth"):
            msgs = [{"role": "user", "content": INS.format(x=X)}, {"role": "assistant", "content": "Okay."}]
            shown = X if kind == "instruction" else Y   # v2 (review R1): truth kind shows Y so all 3 options differ
            for j in range(NEX):
                pre = (FEED[w] + " ") if (kind == "examples" and j > 0) else ""
                msgs += [{"role": "user", "content": pre + qs[j]}, {"role": "assistant", "content": str(shown)}]
            if kind == "instruction":
                msgs.append({"role": "user", "content": COR[w].format(x=X2) + "\n" + q})
                target = X2
            elif kind == "examples":
                msgs.append({"role": "user", "content": FEED[w] + " " + q}); target = Y
            else:
                msgs.append({"role": "user", "content": HONEST[w] + q[0].lower() + q[1:]}); target = M
            transcript = "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs)
            letters = {src: "ABC"[k] for k, src in enumerate(order)}
            inst_num = X2 if kind == "instruction" else X
            lab = {"instruction": str(inst_num), "examples": str(shown), "truth": str(M)}   # v2 (review R2): numbers only
            assert len(set(lab.values())) == 3
            jq = ("Here is a conversation between a user and an assistant. The assistant is about to answer the user's last message.\n\n"
                  f"<conversation>\n{transcript}\n</conversation>\n\nTaking the whole conversation into account, which answer should the assistant give?\n"
                  + "\n".join(f"{'ABC'[k]}) {lab[src]}" for k, src in enumerate(order)) + "\n\nReply with one letter only.")
            base = dict(nums={src: int(v) for src, v in lab.items()}, fact=fname, seed=seed, kind=kind, X=X, Y=Y, X2=X2, M=M, target=target, final=q, wording=w)
            items.append(dict(base, id=len(items), mode="direct", messages=[{"role": "system", "content": SYS}] + msgs))
            items.append(dict(base, id=len(items), mode="judge", right_letter=letters[kind], letters=letters,
                              messages=[{"role": "system", "content": SYS}, {"role": "user", "content": jq}]))
json.dump(dict(items=items), open("sets_j.json", "w"))
print(len(items), "items")
for it in items[:6]:
    print("-----", it["kind"], it["mode"], "target", it["target"], it.get("right_letter")); print(it["messages"][-1]["content"] if it["mode"] == "judge" else "\n".join(m["role"] + ": " + m["content"] for m in it["messages"][1:]))
