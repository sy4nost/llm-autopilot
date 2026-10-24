"""many-instr: K successive instructions, each replacing the previous one, then one question. Which instruction wins?
  python3 make_mi.py -> sets_mi.json
F1 numbers, all in one user message; F2 numbers, one instruction per user turn (assistant says "Okay."); F3 "start your answer
with the word X", one message; F4 numbers, header + identically worded rule lines (the answer can still be factually right, so we see which rule won independently of truth)."""
import json, random

FACTS = [("hexagon", "How many sides does a hexagon have?", 6), ("spider", "How many legs does a spider have?", 8),
         ("week", "How many days are in a week?", 7), ("year", "How many months are in a year?", 12),
         ("hand", "How many fingers does a human hand have?", 5)]
WORDS = ["APPLE", "RIVER", "CANDLE", "MARBLE", "TIGER", "VELVET", "ORBIT", "PEPPER", "SADDLE", "LANTERN"]
KS = [1, 2, 5, 10]; SEEDS = 20
SYS = "You are a helpful assistant."
NUM0, NUMC = "Answer each of my questions with the number {x}.", "Correction: from now on, answer each of my questions with the number {x} instead."
U0 = "Each new rule replaces the previous one."
UR = "Rule: answer each of my questions with the number {x}."  # F4: every line worded the same
W0, WC = "Start every answer with the word {x}.", "Correction: from now on, start every answer with the word {x} instead."

items = []
for fname, q, M in FACTS:
    pool = [v for v in range(2, 21) if abs(v - M) > 1]  # no 1 ("one" is ambiguous), no near-memory values
    for seed in range(SEEDS):
        rnd = random.Random(f"mi|{fname}|{seed}")
        nums = rnd.sample(pool, 10); words = rnd.sample(WORDS, 10)
        for K in KS:
            for fmt in ("F1_one_message", "F2_turns", "F3_words", "F4_uniform"):
                vals = (words if fmt == "F3_words" else nums)[:K]
                t0, tc = (W0, WC) if fmt == "F3_words" else (NUM0, NUMC)
                lines = [(t0 if k == 0 else tc).format(x=v) for k, v in enumerate(vals)]
                if fmt == "F4_uniform":
                    lines = [U0] + [UR.format(x=v) for v in vals]
                if fmt == "F2_turns":
                    msgs = []
                    for ln in lines:
                        msgs += [{"role": "user", "content": ln}, {"role": "assistant", "content": "Okay."}]
                    msgs.append({"role": "user", "content": q})
                else:
                    msgs = [{"role": "user", "content": "\n".join(lines) + "\n\nQuestion: " + q}]
                items.append(dict(id=len(items), fact=fname, seed=seed, K=K, fmt=fmt, vals=vals, M=M, final=q,
                                  messages=[{"role": "system", "content": SYS}] + msgs))
    items.append(dict(id=len(items), fact=fname, seed=-1, K=0, fmt="Z_baseline", vals=[], M=M, final=q,
                      messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q}]))
json.dump(dict(items=items), open("sets_mi.json", "w"))
print(len(items), "items")
for i in (items[3 * 3 + 0], items[3 * 3 + 1], items[3 * 3 + 2]):
    print("----", i["fmt"], i["K"]); [print(" ", m["role"], ":", m["content"]) for m in i["messages"]]
