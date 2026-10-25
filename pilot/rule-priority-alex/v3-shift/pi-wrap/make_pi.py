"""pi-wrap (Alex 2026-10-24 03:00Z): the same counterfactual about pi in three wrappings; options "9" (stated value) and "3" (memory).
python3 -I make_pi.py -> sets_pi.json. 100 values 900..999 (no 3 anywhere in the first digit by construction; first digit 9).
  F  formula:        "Let pi = 974. Q: What is the first digit of pi? A:"
  W  words:          "In this book, the number pi is nine hundred seventy-four. Q: In this book, what is the first digit of pi? A:"
  M  words + digits: "In this book, the number pi is 974. Q: In this book, what is the first digit of pi? A:"
  F2 formula, letter x control (no memory pull): "Let x = 974. Q: What is the first digit of x? A:"
  W2 words, made-up name control: "In this book, the number zorp is nine hundred seventy-four. Q: In this book, what is the first digit of zorp? A:" """
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = [None, None, "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n):
    h, r = divmod(n, 100); s = ONES[h] + " hundred"
    if r == 0:
        return s
    if r < 10:
        return s + " " + ONES[r]
    if r < 20:
        return s + " " + TEENS[r - 10]
    t, o = divmod(r, 10)
    return s + " " + TENS[t] + ("-" + ONES[o] if o else "")


def build():
    vals = random.Random(0).sample(range(900, 1000), 100)
    T = {"F": "Let pi = {n}. Q: What is the first digit of pi? A:",
         "W": "In this book, the number pi is {w}. Q: In this book, what is the first digit of pi? A:",
         "M": "In this book, the number pi is {n}. Q: In this book, what is the first digit of pi? A:",
         "F2": "Let x = {n}. Q: What is the first digit of x? A:",
         "W2": "In this book, the number zorp is {w}. Q: In this book, what is the first digit of zorp? A:"}
    items = []
    for n in vals:
        for k, t in T.items():
            c = ["9", "3"] if len(items) % 2 == 0 else ["3", "9"]   # option order balanced
            items.append(dict(id=len(items), style=k, n=n, prompt=t.format(n=n, w=words(n)), cands=c, key=c.index("9"), habit=c.index("3")))
    return items


if __name__ == "__main__":
    it = build(); json.dump(dict(items=it), open(os.path.join(HERE, "sets_pi.json"), "w"), ensure_ascii=False)
    for k in ("F", "W", "M", "F2", "W2"):
        print(next(x["prompt"] for x in it if x["style"] == k))
    print(len(it))
