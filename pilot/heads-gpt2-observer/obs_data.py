"""Habit-trap items for the GPT-2 'observer floor' pilot (Alex's idea, 2026-10-25).

Task: the classic 'indirect object' sentence that GPT-2 small solves on its own:
    "When Mary and John went to the store, John gave a drink to" -> " Mary" (not " John").
Conditions (same final sentence, only the history before it differs):
  clean    no history.
  trap     history holds the SAME sentence with the roles swapped, already completed:
           "When Mary and John went to the store, Mary gave a drink to John." -> a copy habit pulls toward " John".
  echo     history holds the SAME sentence with the SAME roles, already completed with the right name: an exact duplicate
           exactly like the trap, but copying now gives the right answer (no conflict). The observer must tell trap from echo.
  control  history of the same length and form, about OTHER people (no habit pull toward either answer).
Template sets: A (training) and B (held out, different wording). Names, places and objects are single GPT-2 tokens.
Every item records character spans so the observer can find regions after tokenisation."""
import random

NAMES = ["Mary", "John", "Anna", "Tom", "Kate", "Paul", "Lisa", "Mark", "Emma", "James", "Sarah", "David", "Laura", "Peter",
         "Alice", "Jack", "Rose", "Mike", "Grace", "Adam"]
PLACES = ["store", "park", "school", "office", "station", "garden", "hospital", "market", "library", "beach"]
OBJECTS = ["drink", "book", "ring", "ball", "letter", "key", "gift", "flower", "pen", "cake"]
TEMPLATES = {
    "A": ["When {S} and {IO} went to the {P}, {X} gave a {O} to",
          "After {S} and {IO} arrived at the {P}, {X} handed a {O} to",
          "Then, {S} and {IO} had a long day at the {P}. {X} gave a {O} to"],
    "B": ["While {S} and {IO} were working at the {P}, {X} passed a {O} to",
          "{S} and {IO} met at the {P} yesterday, and {X} sent a {O} to",
          "On the way to the {P}, {S} and {IO} stopped, and {X} threw a {O} to"],
}
# Candidate held-out wordings (2026-10-25: the first B set showed no habit, trap 87/100 vs 81 without history).
# Chosen by step 'select' on its own seed, never on the test seed; rule fixed before the run (PLAN): keep a wording if
# clean >= 85, echo >= 85, no-history >= 85 and trap at least 10 below no-history; take the 3 with the largest trap drop.
B_CANDIDATES = [
    "While {S} and {IO} were working at the {P}, {X} passed a {O} to",
    "{S} and {IO} met at the {P} yesterday, and {X} sent a {O} to",
    "On the way to the {P}, {S} and {IO} stopped, and {X} threw a {O} to",
    "Once {S} and {IO} got to the {P}, {X} brought a {O} to",
    "{S} and {IO} spent the morning at the {P}. Later, {X} offered a {O} to",
    "As {S} and {IO} were leaving the {P}, {X} showed a {O} to",
    "At the {P}, {S} and {IO} talked for hours, and then {X} lent a {O} to",
    "Before {S} and {IO} left the {P}, {X} returned a {O} to",
    "Yesterday {S} and {IO} visited the {P}, where {X} sold a {O} to",
    "{S} and {IO} were cleaning the {P} when {X} tossed a {O} to",
]


def sentence(t, a, b, x, p, o):
    return t.format(S=a, IO=b, X=x, P=p, O=o)


def make_item(r, tset, cond, n_hist):
    t = r.choice(TEMPLATES[tset]); p, o = r.choice(PLACES), r.choice(OBJECTS)
    a, b = r.sample(NAMES, 2)                      # names in mention order; the subject X repeats one of them
    subj = r.choice([a, b]); io = b if subj == a else a
    final = sentence(t, a, b, subj, p, o)
    hist = []
    for k in range(n_hist):
        if cond == "trap" and k == n_hist - 1:     # identical sentence, roles swapped, already answered with the now-wrong name
            hist.append(sentence(t, a, b, io, p, o) + " " + subj + ".")
        elif cond == "echo" and k == n_hist - 1:   # identical sentence, same roles, already answered with the right name
            hist.append(sentence(t, a, b, subj, p, o) + " " + io + ".")
        else:                                      # same form, other people (control, and filler before the trap)
            c, d = r.sample([n for n in NAMES if n not in (a, b)], 2); s2 = r.choice([c, d]); i2 = d if s2 == c else c
            hist.append(sentence(r.choice(TEMPLATES[tset]), c, d, s2, r.choice(PLACES), r.choice(OBJECTS)) + " " + i2 + ".")
    history = (" ".join(hist) + " ") if hist else ""
    text = " " + history + final                   # leading space: every name is the same ' Name' token, even first in text
    return dict(text=text, hist_chars=(1 + len(history)) if hist else 0, final=" " + final, cand=[" " + a, " " + b], target=" " + io, foil=" " + subj,
                cond=cond, tset=tset, n_hist=n_hist)


def build(seed, n_per, tsets=("A", "B"), conds=("clean", "trap", "echo", "control")):
    r = random.Random(seed); out = []
    for ts in tsets:
        for c in conds:
            for _ in range(n_per):
                out.append(make_item(r, ts, c, 0 if c == "clean" else r.choice([1, 2, 3])))
    r.shuffle(out)                                 # mixed batches: no batch is a single condition
    return out
