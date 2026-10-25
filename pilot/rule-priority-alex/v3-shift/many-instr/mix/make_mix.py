"""many-instr/mix: 5 successive instructions (one per user turn, assistant "Okay."; the LAST is the valid one), then N past
answers of the assistant to other questions, then the test question. Arms: examples follow the FIRST instruction (E1), the
PENULTIMATE instruction (E4), or a NEW number that no instruction named (EX).  python3 make_mix.py -> sets_mix.json"""
import ast, json, os, random
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "three-forces", "make_tf.py")).read()
BANK = ast.literal_eval(_src[_src.index("BANK = ") + 7:_src.index("}", _src.index("BANK = ")) + 1])  # same question bank, no import side effects
FACTS = [("hexagon", "How many sides does a hexagon have?", 6), ("spider", "How many legs does a spider have?", 8),
         ("week", "How many days are in a week?", 7), ("year", "How many months are in a year?", 12),
         ("hand", "How many fingers does a human hand have?", 5)]
NS = [0, 1, 2, 3, 5, 8]; SEEDS = 20; K = 5
SYS = "You are a helpful assistant."
NUM0, NUMC = "Answer each of my questions with the number {x}.", "Correction: from now on, answer each of my questions with the number {x} instead."
TOPIC = {"week": ["How many hours are in a week?"], "year": ["How many weeks are in a year?"], "hexagon": [], "spider": [], "hand": []}

items = []
for fname, q, M in FACTS:
    pool = [v for v in range(2, 21) if abs(v - M) > 1]
    for seed in range(SEEDS):
        rnd = random.Random(f"mix|{fname}|{seed}")
        drawn = rnd.sample(pool, K + 1); vals, x = drawn[:K], drawn[K]
        bank = [b for b, a in BANK.items() if b not in TOPIC[fname] and a not in drawn and abs(a - M) > 1 and "hexagon" not in b]
        qs = rnd.sample(bank, max(NS))
        head = []
        for k, v in enumerate(vals):
            head += [{"role": "user", "content": (NUM0 if k == 0 else NUMC).format(x=v)}, {"role": "assistant", "content": "Okay."}]
        for arm, ev in (("E1_first", vals[0]), ("E4_penultimate", vals[3]), ("EX_new", x)):
            for N in NS:
                if N == 0 and arm != "E1_first":
                    continue
                msgs = list(head)
                for j in range(N):
                    msgs += [{"role": "user", "content": qs[j]}, {"role": "assistant", "content": str(ev)}]
                msgs.append({"role": "user", "content": q})
                items.append(dict(id=len(items), fact=fname, seed=seed, arm=arm if N else "N0", N=N, vals=vals, ex=ev if N else None,
                                  M=M, final=q, messages=[{"role": "system", "content": SYS}] + msgs))
json.dump(dict(items=items), open("sets_mix.json", "w"))
print(len(items), "items")
it = [i for i in items if i["arm"] == "E4_penultimate" and i["N"] == 2][0]
[print(" ", m["role"], ":", m["content"]) for m in it["messages"]]
