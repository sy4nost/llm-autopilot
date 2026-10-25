"""forces-map: does a "map of forces" step (the model fills three lines: what the user's instructions say, what the
assistant's earlier answers suggest, what general knowledge says) work as a lamp that fires on traps and stays quiet on
calm dialogues? (Alex, 2026-10-24: "Well, let's try. Run it.")
Self-built mixed set, 2 families x 4 cells x 30 = 240 dialogues:
  NUM : "Answer each of my questions with the number X." / "Okay." / Q1 -> X / [change: "From now on ... number Z
        instead." / "Okay."] / Q2..Q4 -> S / final fact question (true answer M).
  PREF: user sets a preference A / filler / [change: "Actually, change my <slot> to B."] / three assistant turns that
        state "your <slot> is S" / final "What is my current <slot>?".
  cells: calm_keep (no change, S = right), calm_change (change, S = right), trap_keep (no change, S = another value),
         trap_change (change, S = the old value). Right answer = the user's latest instruction (X / Z / A / B).
  python3 make_fm.py base   -> sets_d.json (DIRECT), sets_th.json (THINK), sets_map.json (MAP step), sets_g1.json (GOAL step)"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "..", "three-forces", "make_tf.py")).read()
BANK = ast.literal_eval(_src[_src.index("BANK = ") + 7:_src.index("}", _src.index("BANK = ")) + 1])
FACTS = [("hexagon", "How many sides does a hexagon have?", 6), ("spider", "How many legs does a spider have?", 8),
         ("week", "How many days are in a week?", 7), ("year", "How many months are in a year?", 12),
         ("hand", "How many fingers does a human hand have?", 5)]
EXCL = {"spider": ["legs"], "hexagon": ["sides"], "week": ["week"], "year": ["year"], "hand": []}
SLOTS = {"coffee drink": ["latte", "espresso", "cappuccino", "americano", "mocha", "macchiato"],
         "music genre": ["jazz", "rock", "blues", "reggae", "techno", "opera"],
         "color theme": ["red", "blue", "green", "orange", "purple", "yellow"]}
FILL = [("Can you remind me how these settings work?", "Sure, just tell me what you'd like and I'll update it."),
        ("Is there anything else I should know?", "Not really, you can change your settings at any time."),
        ("Thanks, that's helpful.", "You're welcome! Let me know if you need anything else."),
        ("What else can you help me with?", "I can help with planning, suggestions and your other settings."),
        ("Sounds good.", "Great, I'm here whenever you need me.")]
TIP_U = ["Can you suggest something for this weekend?", "Any idea for this evening?", "What would you recommend for tomorrow?"]
TIP_A = ["Since your {s} is {v}, I'd suggest planning something relaxing around it.",
         "With your {s} set to {v}, a good idea would be to share it with a friend.",
         "Your {s} is {v}, so you could try something new that fits it."]
CELLS = ["calm_keep", "calm_change", "trap_keep", "trap_change"]; PER = 30
SYS = "You are a helpful assistant."
THINK = ("\n\nBefore answering, think step by step: go through the conversation in order and work out what answer "
         "my instructions call for now (if I changed an instruction, the latest one counts). Then give the final answer on the last line as \"Answer: <answer>\".")
INTRO = ("Here is a conversation between a user and an assistant. The assistant is about to answer the user's last "
         "message.\n\n<conversation>\n{t}\n</conversation>\n\nDo not answer the user's question. ")
MAPQ = INTRO + ("Instead, fill in exactly three lines. Each line must contain only one short value (a number or a word), "
                "or \"none\".\nLine 1: the answer the user's own instructions call for (if the user changed an instruction, "
                "the latest one).\nLine 2: the answer the assistant's most recent replies in this conversation gave.\n"
                "Line 3: the answer general knowledge would give, ignoring the conversation.\nWrite them as:\n"
                "USER'S INSTRUCTIONS: ...\nASSISTANT'S RECENT REPLIES: ...\nGENERAL KNOWLEDGE: ...")
# v1.2 after smoke 2: placeholders in <...> were echoed (OLMo) and line 1 was read as "quote the user's message" (phi)
GQ = INTRO + ("Instead, say in one sentence what the user wants from the assistant's next answer right now, taking the whole "
              "conversation into account, and name the specific value the user expects the answer to contain.")


def num_item(rnd, cell):
    fname, q, M = rnd.choice(FACTS)
    pool = [v for v in range(2, 21) if abs(v - M) > 1]
    X, Z, Y = rnd.sample(pool, 3)
    bank = [b for b, a in BANK.items() if a not in (X, Y, Z) and abs(a - M) > 1 and not any(w in b for w in EXCL[fname])]
    qs = rnd.sample(bank, 4); change = cell.endswith("change")
    right = Z if change else X
    S = right if cell.startswith("calm") else (X if change else Y)
    msgs = [("user", f"Answer each of my questions with the number {X}."), ("assistant", "Okay."), ("user", qs[0]), ("assistant", str(X))]
    if change:
        msgs += [("user", f"From now on, answer each of my questions with the number {Z} instead."), ("assistant", "Okay.")]
    for j in range(1, 4):
        msgs += [("user", qs[j]), ("assistant", str(S))]
    msgs.append(("user", q))
    return dict(fam="NUM", right=right, shown=S, old=X if change else None, M=M, final=q, values=None, msgs=msgs)


def pref_item(rnd, cell):
    slot = rnd.choice(sorted(SLOTS)); A, B, C = rnd.sample(SLOTS[slot], 3); change = cell.endswith("change")
    right = B if change else A
    S = right if cell.startswith("calm") else (A if change else C)
    msgs = [("user", f"I'd like to set up my {slot} preference."), ("assistant", "Sure, what would you like?"),
            ("user", f"Set my {slot} to {A}."), ("assistant", f"Done, your {slot} is now {A}.")]
    f1, f2 = rnd.sample(FILL, 2)
    msgs += [("user", f1[0]), ("assistant", f1[1])]
    if change:
        msgs += [("user", f"Actually, change my {slot} to {B}."), ("assistant", f"Done, your {slot} is now {B}.")]
    else:
        msgs += [("user", f2[0]), ("assistant", f2[1])]
    for j in range(3):
        msgs += [("user", TIP_U[j]), ("assistant", TIP_A[j].format(s=slot, v=S))]
    msgs.append(("user", f"What is my current {slot}? Reply with the value only."))
    return dict(fam="PREF", right=right, shown=S, old=A if change else None, M=None, final=msgs[-1][1], values=SLOTS[slot], msgs=msgs)


def build():
    items = []
    for fam, f in (("NUM", num_item), ("PREF", pref_item)):
        for cell in CELLS:
            for seed in range(PER):
                it = f(random.Random(f"fm|{fam}|{cell}|{seed}"), cell)
                it.update(id=f"{fam}|{cell}|{seed}", cell=cell, seed=seed)
                it["messages"] = [{"role": "system", "content": SYS}] + [{"role": r, "content": c} for r, c in it.pop("msgs")]
                items.append(it)
    return items


def transcript(msgs):
    return "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs if m["role"] != "system")


ITEMS = build()
if __name__ == "__main__" and sys.argv[1:] == ["base"]:
    d, th, mp, g1 = [], [], [], []
    for it in ITEMS:
        d.append(dict(id=it["id"], messages=it["messages"]))
        last = dict(it["messages"][-1]); last["content"] += THINK
        th.append(dict(id=it["id"], messages=it["messages"][:-1] + [last]))
        t = transcript(it["messages"]); s = [{"role": "system", "content": SYS}]
        mp.append(dict(id=it["id"], messages=s + [{"role": "user", "content": MAPQ.format(t=t)}]))
        g1.append(dict(id=it["id"], messages=s + [{"role": "user", "content": GQ.format(t=t)}]))
    for n, x in (("d", d), ("th", th), ("map", mp), ("g1", g1)):
        json.dump(dict(items=x), open(os.path.join(HERE, f"sets_{n}.json"), "w"))
    print(len(d), "dialogues")
    for k in (0, 45, 150, 225):
        print("----", ITEMS[k]["id"], "right", ITEMS[k]["right"]); print(transcript(ITEMS[k]["messages"]))
    print("----MAP prompt tail:\n" + mp[0]["messages"][-1]["content"][-520:])
