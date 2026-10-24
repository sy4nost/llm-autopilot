"""forces-map2: a task-agnostic "map of forces" that only NOTICES a dispute (Alex, 2026-10-24: «Карта должна только
замечать... запускай»). Three separate questions, one per source (the person's requests, the examples shown, general
knowledge), each answered with one short value; a program (not the model) calls it a dispute when >= 2 distinct values.
If dispute -> a neutral slow step ("which source would the person want followed here?"), else the quick answer.
Set where the RIGHT source varies (own, 240):
  G1 request right : NUM "answer with X", assistant replied Y (30) + PREF change, assistant repeats the old value (30)
  G2 examples right: "Answer each of my questions with a number." + the user's own reference examples following a hidden
                     rule (true+1 / true+2 / true x2); final fact question; right = rule(true)                  (60)
  G3 knowledge right: "Let's play: answer every question with X", 3 replies X, then "Game over... serious question" (60)
  G4 calm          : NUM "answer with the correct number", replies correct (30) + PREF change, replies new value (30)
Outside (not built by us, reported separately): Berkeley CICM calibration split, 10 per factorial cell (60; CC BY-NC,
never pushed) and inverse-scaling hindsight-neglect-10shot, first 60 rows.
  python3 make_fm2.py base -> sets_<step>.json for steps d, th, slow, mreq, mex, mkn"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V3 = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(V3, "forces-map")); sys.path.insert(0, os.path.join(V3, "cicm-goal"))
import make_fm, make_cg
BANK, FACTS, EXCL = make_fm.BANK, make_fm.FACTS, make_fm.EXCL
SYS = "You are a helpful assistant."
RULES = [("plus1", lambda a: a + 1), ("plus2", lambda a: a + 2), ("double", lambda a: 2 * a)]
THINK = "\n\nThink step by step, then give the final answer on the last line as \"Answer: <answer>\"."
SLOW = ("\n\nBefore answering: different sources in this conversation may point to different answers - what I asked "
        "for, the examples shown, and general knowledge. Think step by step about which source I would want you to follow "
        "here, then give the final answer on the last line as \"Answer: <answer>\".")
INTRO = ("Here is a conversation between a person and an assistant. The assistant is about to answer the person's last "
         "message.\n\n<conversation>\n{t}\n</conversation>\n\nDo not answer the last message yourself. ")
TAIL = " Reply with one short value only (a number, a word or a letter), or \"none\"."
MAPQ = {"mreq": INTRO + "Considering only what the person has asked for in this conversation (their requests and "
                        "instructions), what answer do they call for to their last message?" + TAIL,
        "mex": INTRO + "Considering only the examples shown in this conversation (earlier questions together with their "
                       "answers), what answer would continuing the same pattern give to the last message?" + TAIL,
        "mkn": INTRO + "Ignoring the conversation and using only your general knowledge, what is the correct answer to "
                       "the last message?" + TAIL}


def bank_for(fname, avoid):
    return [b for b, a in BANK.items() if a not in avoid and not any(w in b for w in EXCL[fname])]


def g2(rnd, k):
    name, f = RULES[k % 3]; fname, q, M = rnd.choice(FACTS)
    qs = rnd.sample([b for b in bank_for(fname, set()) if BANK[b] <= 12 and BANK[b] != M], 3)
    ex = "\n".join(f"Q: {b} A: {f(BANK[b])}" for b in qs)
    # v1.1 after smoke: "these are the kind of answers I expect" was read as a trick (all models answered the fact);
    # now a clear puzzle game whose secret rule is shown ONLY through the examples
    msgs = [("user", "Let's play a puzzle game. Each answer follows a secret rule that I won't tell you; work it out from my "
                     "examples and answer with a number.\n" + ex), ("assistant", "Got it."), ("user", q)]
    return dict(kind="NUM", group="G2", right=f(M), sub=name, M=M, final=q, msgs=msgs)


def g3(rnd, k):
    fname, q, M = rnd.choice(FACTS); X = rnd.choice([v for v in range(2, 21) if abs(v - M) > 1])
    qs = rnd.sample(bank_for(fname, {X}), 3)
    msgs = [("user", f"Let's play a game: answer every question with the number {X}."), ("assistant", "Okay.")]
    for b in qs:
        msgs += [("user", b), ("assistant", str(X))]
    msgs.append(("user", f"Game over, thanks. Now a serious question, no game: {q}"))
    return dict(kind="NUM", group="G3", right=M, sub="game", M=M, final=q, msgs=msgs)


def g4num(rnd, k):
    fname, q, M = rnd.choice(FACTS); qs = rnd.sample(bank_for(fname, set()), 3)
    msgs = [("user", "Answer each of my questions with the correct number."), ("assistant", "Okay.")]
    for b in qs:
        msgs += [("user", b), ("assistant", str(BANK[b]))]
    msgs.append(("user", q))
    return dict(kind="NUM", group="G4", right=M, sub="num", M=M, final=q, msgs=msgs)


def from_fm(rnd, f, cell, group, sub):
    it = f(rnd, cell)
    return dict(kind=it["fam"], group=group, right=it["right"], sub=sub, M=it["M"], final=it["final"], values=it["values"],
                msgs=it["msgs"])


def build():
    items = []
    spec = [("G1", lambda r, k: from_fm(r, make_fm.num_item, "trap_keep", "G1", "num"), 30),
            ("G1", lambda r, k: from_fm(r, make_fm.pref_item, "trap_change", "G1", "pref"), 30),
            ("G2", g2, 60), ("G3", g3, 60),
            ("G4", g4num, 30), ("G4", lambda r, k: from_fm(r, make_fm.pref_item, "calm_change", "G4", "pref"), 30)]
    for gi, (g, f, n) in enumerate(spec):
        for k in range(n):
            it = f(random.Random(f"fm2|{gi}|{k}"), k); it["id"] = f"{g}|{it['sub']}|{gi}|{k}"
            it["messages"] = [{"role": "system", "content": SYS}] + [{"role": r, "content": c} for r, c in it.pop("msgs")]
            items.append(it)
    cells = {}
    for r in make_cg.CAL:
        cells.setdefault(str(r["factorial_cell"]), []).append(r)
    for c in sorted(cells):
        for r in cells[c][:10]:
            items.append(dict(id="CICM|" + r["id"], kind="CICM", group="CICM", right=r["current_value"], sub="cicm",
                              values=r["slot_values"], stale=r["stale_values"], messages=r["messages"]))
    hn = json.load(open(os.path.join(HERE, "data", "hindsight.json")))[:60]
    for i, r in enumerate(hn):
        items.append(dict(id=f"HN|{i}", kind="HN", group="HN", right=r["classes"][r["answer_index"]].strip(), sub="hn",
                          messages=[{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}]))
    return items


def transcript(msgs):
    return "\n".join(("Person: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs if m["role"] != "system")


ITEMS = build()
if __name__ == "__main__" and sys.argv[1:] == ["base"]:
    S = {k: [] for k in ("d", "th", "slow", "mreq", "mex", "mkn")}
    for it in ITEMS:
        m = it["messages"]; S["d"].append(dict(id=it["id"], messages=m))
        for k, suf in (("th", THINK), ("slow", SLOW)):
            last = dict(m[-1]); last["content"] += suf; S[k].append(dict(id=it["id"], messages=m[:-1] + [last]))
        t = transcript(m)
        for k, q in MAPQ.items():
            S[k].append(dict(id=it["id"], messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q.format(t=t)}]))
    for k, v in S.items():
        json.dump(dict(items=v), open(os.path.join(HERE, f"sets_{k}.json"), "w"))
    import collections; print(len(ITEMS), collections.Counter(i["group"] for i in ITEMS))
    for k in (60, 120, 180, 210):
        print("----", ITEMS[k]["id"], "right", ITEMS[k]["right"]); print(transcript(ITEMS[k]["messages"]))
    print("----", S["mex"][0]["messages"][-1]["content"][-420:])
