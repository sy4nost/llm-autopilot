"""hex-wrap (Alex 2026-10-24 03:10Z): stated count contradicting the real one, as a formula vs as a story.
Options: stated (right), real count (memory), another number. python3 -I make_hex.py -> sets_pi.json (same runner as pi-wrap)
  B  story:   "In this book, a 'hexagon' has 4 sides. Q: In this book, how many sides does a hexagon have? A:"
  F  formula: "Let sides(hexagon) = 4. Q: What is sides(hexagon)? A:"
  B2 story, made-up word:   "In this book, a 'zorbagon' has 4 sides. Q: In this book, how many sides does a zorbagon have? A:"
  F2 formula, made-up word: "Let sides(zorbagon) = 4. Q: What is sides(zorbagon)? A:" """
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
OBJ = [("hexagon", "sides", 6), ("octagon", "sides", 8), ("pentagon", "sides", 5), ("triangle", "sides", 3), ("square", "sides", 4),
       ("cube", "faces", 6), ("spider", "legs", 8), ("dog", "legs", 4), ("ant", "legs", 6), ("octopus", "arms", 8),
       ("starfish", "arms", 5), ("hand", "fingers", 5), ("bicycle", "wheels", 2), ("tricycle", "wheels", 3), ("week", "days", 7)]
FAKE = "zorbagon blint kravid mupple quorle sneet trabbit vonk wibbet yarlix glimp drosk fennix plorb skeem".split()


def an(w):
    return "an" if w[0] in "aeiou" else "a"


def build():
    rng = random.Random(0); items = []
    for (o, part, real), fake in zip(OBJ, FAKE):
        for stated in rng.sample([k for k in range(2, 13) if k != real], 6):
            other = rng.choice([k for k in range(2, 13) if k not in (real, stated)])
            c = [str(stated), str(real), str(other)]; rng.shuffle(c)
            for st, w in (("B", o), ("F", o), ("B2", fake), ("F2", fake)):
                p = (f"In this book, {an(w)} '{w}' has {stated} {part}. Q: In this book, how many {part} does {an(w)} {w} have? A:" if st.startswith("B")
                     else f"Let {part}({w}) = {stated}. Q: What is {part}({w})? A:")
                items.append(dict(id=len(items), style=st, n=f"{o}:{stated}", prompt=p, cands=c, key=c.index(str(stated)), habit=c.index(str(real))))
    return items


if __name__ == "__main__":
    it = build(); json.dump(dict(items=it), open(os.path.join(HERE, "sets_pi.json"), "w"), ensure_ascii=False)
    for k in ("B", "F", "B2", "F2"):
        print(next(x["prompt"] for x in it if x["style"] == k))
    print(len(it))
