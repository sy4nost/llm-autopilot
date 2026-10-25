"""Far-transfer items for the observer floor: 'what comes after' (numbers, days, months, letters).
Same shape of habit as the names task, different task: the history ends with the token that now starts the question.
  clean    "The day after Monday is" -> " Tuesday"
  trap     "The day after Sunday is Monday. The day after Monday is" -> " Tuesday" (copy habit pulls to " Monday")
  echo     "The day after Monday is Tuesday. The day after Monday is" -> " Tuesday" (copying is right)
  control  history about other members of the same sequence (no pull toward either candidate)
Candidates: the asked token (foil, it is repeated in the question) and the right next token, in random order."""
import random

SEQS = {
    "number": ("The number after {X} is", [str(i) for i in range(1, 21)]),
    "day": ("The day after {X} is", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    "month": ("The month after {X} is", ["January", "February", "March", "April", "May", "June", "July", "August",
                                         "September", "October", "November", "December"]),
    "letter": ("The letter after {X} is", list("ABCDEFGHIJKLMNOPQRSTUVWXY")),
}


def nxt(seq, i):
    return seq[(i + 1) % len(seq)]


def make_item(r, kind, cond, n_hist):
    t, seq = SEQS[kind]
    i = r.randrange(len(seq) - 1)
    x, y = seq[i], seq[i + 1]                                   # question: after x -> y
    prev = seq[i - 1] if i > 0 else seq[-1]
    final = t.format(X=x)
    far = [k for k in range(len(seq) - 1) if abs(k - i) > 2]
    hist = []
    for k in range(n_hist):
        if cond == "trap" and k == n_hist - 1:
            hist.append(t.format(X=prev) + " " + x + ".")       # ends with x, the token asked about now
        elif cond == "echo" and k == n_hist - 1:
            hist.append(t.format(X=x) + " " + y + ".")
        else:
            j = r.choice(far); hist.append(t.format(X=seq[j]) + " " + seq[j + 1] + ".")
    history = (" ".join(hist) + " ") if hist else ""
    cand = [" " + x, " " + y]; r.shuffle(cand)
    return dict(text=" " + history + final, hist_chars=(1 + len(history)) if hist else 0, final=" " + final, cand=cand,
                target=" " + y, foil=" " + x, cond=cond, tset="far_" + kind, n_hist=n_hist)


def build(seed, n_per, kinds=tuple(SEQS), conds=("clean", "trap", "echo", "control")):
    r = random.Random(seed); out = []
    for kd in kinds:
        for c in conds:
            for _ in range(n_per):
                out.append(make_item(r, kd, c, 0 if c == "clean" else r.choice([1, 2, 3])))
    r.shuffle(out)
    return out
