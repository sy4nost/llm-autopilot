"""cycle3: instruction (I), shown examples (E) and truth (T) on the SAME questions, every pair in one test.
  python3 make_c3.py -> sets_c3.json
Format for every arm: instruction lines as separate user turns (assistant "Okay."), then N past Q/A turns, then the question."""
import ast, json, os, random
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "three-forces", "make_tf.py")).read()
BANK = ast.literal_eval(_src[_src.index("BANK = ") + 7:_src.index("}", _src.index("BANK = ")) + 1])
FACTS = [("hexagon", "How many sides does a hexagon have?", 6, []), ("spider", "How many legs does a spider have?", 8, ["How many legs does an insect have?", "How many legs does a dog have?", "How many legs does a starfish usually have?"]),
         ("week", "How many days are in a week?", 7, ["How many hours are in a week?"]), ("year", "How many months are in a year?", 12, ["How many weeks are in a year?"]),
         ("hand", "How many fingers does a human hand have?", 5, []), ("pi", "What is the value of pi, to two decimal places?", 3.14, [])]
SEEDS = 20
SYS = "You are a helpful assistant."
INS = "Answer each of my questions with the number {x}."
COR = "Correction: from now on, answer each of my questions with the number {x} instead."
GEN = "Answer each of my questions with a single number."  # the 'number game' without naming a number
NEU = "Let's start."  # control preface: same extra user turn + "Okay.", no number game (review R: separates content from format)


def turns(lines):
    m = []
    for ln in lines:
        m += [{"role": "user", "content": ln}, {"role": "assistant", "content": "Okay."}]
    return m


items = []
for fname, q, M, excl in FACTS:
    pi = isinstance(M, float)
    pool = [4.25, 2.86, 5.17, 1.73] if pi else [v for v in range(2, 21) if abs(v - M) > 1]
    for seed in range(SEEDS):
        rnd = random.Random(f"c3|{fname}|{seed}")
        X, Y, X2 = rnd.sample(pool, 3)
        bank = [b for b, a in BANK.items() if b not in excl and a not in (X, Y, X2) and (pi or abs(a - M) > 1)
                and not (fname == "hexagon" and "sides" in b)]
        qs = rnd.sample(bank, 4)
        ex = lambda n, y: sum(([{"role": "user", "content": qs[j]}, {"role": "assistant", "content": str(y)}] for j in range(n)), [])
        arms = {"T_alone": [], "IT_instr": turns([INS.format(x=X)]),
                "II_correction": turns([INS.format(x=X), COR.format(x=X2)])}
        for n in (2, 4):
            arms[f"ET_examples_N{n}"] = ex(n, Y)
            arms[f"GE_game_examples_N{n}"] = turns([GEN]) + ex(n, Y)
            arms[f"CE_neutral_examples_N{n}"] = turns([NEU]) + ex(n, Y)
            arms[f"IE_instr_examples_N{n}"] = turns([INS.format(x=X)]) + ex(n, Y)
        for arm, body in arms.items():
            items.append(dict(id=len(items), fact=fname, seed=seed, arm=arm, X=X, Y=Y, X2=X2, M=M, final=q,
                              messages=[{"role": "system", "content": SYS}] + body + [{"role": "user", "content": q}]))
json.dump(dict(items=items), open("sets_c3.json", "w"))
print(len(items), "items")
for it in items[:9]:
    print("---", it["arm"], "X", it["X"], "Y", it["Y"], "X2", it["X2"]); [print("  ", m["role"], ":", m["content"]) for m in it["messages"][1:]]
