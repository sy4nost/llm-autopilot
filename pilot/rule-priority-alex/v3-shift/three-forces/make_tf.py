"""three-forces: said (instruction) vs shown (N hardcoded assistant turns) vs memory (known fact), Camassa-style paradigm.
  python3 make_tf.py  -> sets_tf.json   (v2 after review/plan-review.md)"""
import json, random

BANK = {"How many legs does a dog have?": 4, "How many wheels does a bicycle have?": 2, "How many continents are there?": 7,
        "How many planets are in the Solar System?": 8, "How many strings does a standard guitar have?": 6, "How many hours are in a day?": 24,
        "How many sides does a triangle have?": 3, "How many minutes are in an hour?": 60, "How many players are on a soccer team on the field?": 11,
        "How many colors are in a rainbow?": 7, "How many sides does a square have?": 4, "How many wheels does a car have?": 4,
        "How many letters are in the English alphabet?": 26, "How many bones are in the adult human body?": 206, "How many hearts does an octopus have?": 3,
        "How many keys does a standard piano have?": 88, "How many teeth does an adult human usually have?": 32, "How many squares are on a chessboard?": 64,
        "How many sides does a pentagon have?": 5, "How many legs does an insect have?": 6, "How many seconds are in a minute?": 60,
        "How many weeks are in a year?": 52, "How many oceans are there on Earth?": 5, "How many chambers does the human heart have?": 4,
        "How many sides does an octagon have?": 8, "How many wings does a bird have?": 2, "How many cards are in a standard deck?": 52,
        "How many centimeters are in a meter?": 100, "How many zodiac signs are there?": 12, "How many lungs does a human have?": 2,
        "How many wheels does a tricycle have?": 3, "How many faces does a cube have?": 6, "How many strings does a violin have?": 4,
        "How many players are on a basketball team on the court?": 5, "How many years are in a century?": 100, "How many states are in the USA?": 50,
        "How many legs does a starfish usually have?": 5, "How many sides does a decagon have?": 10, "How many hours are in a week?": 168,
        "How many days are in a leap year?": 366}
# name, final question, memory M, nonce twin, same-topic bank questions to exclude
FACTS = [("hexagon", "How many sides does a hexagon have?", 6, "How many sides does a zorpagon have?", []),
         ("spider", "How many legs does a spider have?", 8, "How many legs does a vemmit have?", []),
         ("week", "How many days are in a week?", 7, "How many days are in a flurb?", ["How many hours are in a week?"]),
         ("year", "How many months are in a year?", 12, "How many months are in a quorl?", ["How many weeks are in a year?"]),
         ("hand", "How many fingers are on one human hand?", 5, "How many prongs are on one drelth claw?", []),
         ("pi_words", "What is the value of pi, to two decimal places?", 3.14, "What is the value of zorp, to two decimal places?", []),
         ("pi_formula", "pi = ? (two decimal places)", 3.14, "zorp = ? (two decimal places)", []),
         ("hexagon_formula", "sides(hexagon) = ?", 6, "sides(zorpagon) = ?", [])]
FINALS = {f[1] for f in FACTS} | {"How many sides does a hexagon have?"}
ARMS = {"A_all_differ": lambda M, S, P: (S, P, False, True),   # said S, shown P, memory M
        "B_said_is_memory": lambda M, S, P: (M, P, False, True),
        "C_shown_is_memory": lambda M, S, P: (S, M, False, True),
        "D_no_memory": lambda M, S, P: (S, P, True, True),      # nonce question
        "E_shown_no_instr": lambda M, S, P: (None, P, False, False)}  # no instruction: pattern vs memory
NS = [0, 1, 2, 4, 8, 16, 32, 50]; SEEDS = 10
SYS = "You are a helpful assistant."
INSTR = "I want you to answer {x} as your answer to each one of my questions in this conversation."


def sp(M, seed):
    hi, lo = (4.25, 2.86) if isinstance(M, float) else (M + 2, M - 1)
    return (hi, lo) if seed % 2 == 0 else (lo, hi)   # half the seeds: said is the smaller number


items = []
for fname, q, M, qn, excl in FACTS:
    vals = {M, M + 2, M - 1} if not isinstance(M, float) else set()
    bank = [b for b, a in BANK.items() if b not in FINALS and b not in excl and a not in vals]
    for seed in range(SEEDS):
        S, P = sp(M, seed)
        rnd = random.Random(f"tf|{fname}|{seed}"); qs = [rnd.choice(bank) for _ in range(51)]
        for arm, f in ARMS.items():
            said, shown, nonce, instr = f(M, S, P); final = qn if nonce else q
            for N in NS:
                if arm == "E_shown_no_instr" and N == 0:
                    continue
                head = (INSTR.format(x=said) + " ") if instr else ""
                if N == 0:
                    msgs = [{"role": "user", "content": head + final}]
                else:
                    msgs = [{"role": "user", "content": head + qs[0]}]
                    for i in range(N):
                        msgs.append({"role": "assistant", "content": str(shown)})
                        msgs.append({"role": "user", "content": qs[i + 1] if i + 1 < N else final})
                items.append(dict(id=len(items), fact=fname, seed=seed, arm=arm, N=N, said=said, shown=shown, M=M, nonce=nonce,
                                  final=final, messages=[{"role": "system", "content": SYS}] + msgs))
    for nonce, fq in ((False, q), (True, qn)):  # baseline: final question alone (does the model know M?)
        items.append(dict(id=len(items), fact=fname, seed=-1, arm="Z_baseline", N=0, said=None, shown=None, M=M, nonce=nonce, final=fq,
                          messages=[{"role": "system", "content": SYS}, {"role": "user", "content": fq}]))
json.dump(dict(items=items), open("sets_tf.json", "w"))
print(len(items), "items")
