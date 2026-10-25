"""Own item set for the conflict-finding training test (PLAN.md).  python3 -I make_ctr.py SEED  -> own_items.jsonl
Each item is a conversation (list of chat messages, last turn = the user's request) plus the target line:
  Conflict: yes|no. Between: <source> and <source>. Why: <one sentence>.
A conflict = two parts of the context call for different answers to the final request. Every family has decoys built from
the same parts and words in which all parts agree (label 'no').
Train families (T1-T4) and held-out own families (H1, H3) use disjoint domains, names and wordings."""
import json, os, random, sys, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, N_DEV, N_HELD = 400, 50, 200          # per family; half conflict, half decoy

NAMES = ["Maya", "Omar", "Lena", "Tom", "Priya", "Jonas", "Aiko", "Diego", "Sara", "Felix", "Nadia", "Ivan",
         "Chloe", "Kwame", "Rosa", "Ben", "Hana", "Luca", "Zoe", "Amir"]
SYS = ["You are a helpful assistant.", "You are a friendly AI assistant.", "You are an assistant that helps with everyday tasks.", None]


def sys_msgs(r):
    s = r.choice(SYS)
    return [{"role": "system", "content": s}] if s else []


def why(r, opts):
    return r.choice(opts)


def target(conf, a, b, w):
    return f"Conflict: {'yes' if conf else 'no'}. Between: {a} and {b}. Why: {w}"


# ---------- T1 rule changed after worked examples ----------
RULES = [  # (domain, item noun, attribute name, values, old rule fn, change text, new rule fn, label names)
    dict(dom="parcel shipping", noun="parcel", attrs=[("weight_kg", list(range(1, 31))), ("express", [True, False])],
         old=lambda a: "LARGE" if a["weight_kg"] > 10 else "SMALL",
         chg=["From now on the limit is 20 kg instead of 10 kg.", "Update: parcels count as LARGE only above 20 kg now.",
              "New policy starting today: the weight threshold moves from 10 kg to 20 kg."],
         new=lambda a: "LARGE" if a["weight_kg"] > 20 else "SMALL",
         same=["Reminder: the limit is still 10 kg.", "Note: nothing changes, LARGE still means above 10 kg."],
         fmt=lambda a: f"weight {a['weight_kg']} kg, express: {'yes' if a['express'] else 'no'}",
         rule="A parcel is LARGE if it weighs more than 10 kg, otherwise SMALL."),
    dict(dom="customer support tickets", noun="ticket", attrs=[("hours_open", list(range(1, 73))), ("vip", [True, False])],
         old=lambda a: "URGENT" if a["vip"] else "NORMAL",
         chg=["Change: from now on a ticket is URGENT only if it is from a VIP AND has been open more than 24 hours.",
              "Update to the rule: VIP tickets are URGENT only after 24 hours open.",
              "Starting now, URGENT requires both VIP status and more than 24 open hours."],
         new=lambda a: "URGENT" if a["vip"] and a["hours_open"] > 24 else "NORMAL",
         same=["Just to confirm: every VIP ticket is URGENT.", "No change: VIP means URGENT."],
         fmt=lambda a: f"open {a['hours_open']} hours, VIP: {'yes' if a['vip'] else 'no'}",
         rule="A ticket is URGENT if the customer is a VIP, otherwise NORMAL."),
    dict(dom="exam grading", noun="student", attrs=[("score", list(range(30, 101))), ("retake", [True, False])],
         old=lambda a: "PASS" if a["score"] >= 50 else "FAIL",
         chg=["New rule from today: the pass mark is 70, not 50.", "Correction to the policy: you now need 70 points to pass.",
              "The department raised the pass mark to 70."],
         new=lambda a: "PASS" if a["score"] >= 70 else "FAIL",
         same=["Remember, the pass mark stays at 50.", "No update: 50 points is still a pass."],
         fmt=lambda a: f"score {a['score']}, retake: {'yes' if a['retake'] else 'no'}",
         rule="A student PASSES with a score of 50 or more, otherwise FAILS."),
    dict(dom="loyalty bonus", noun="order", attrs=[("amount", list(range(10, 301, 5))), ("member", [True, False])],
         old=lambda a: "BONUS" if a["amount"] >= 100 else "NONE",
         chg=["From now on members get a BONUS from 50 dollars; non-members still need 100.",
              "Policy change: the bonus threshold for members drops to 50 dollars.",
              "Update: members qualify for BONUS at 50 dollars or more."],
         new=lambda a: "BONUS" if a["amount"] >= (50 if a["member"] else 100) else "NONE",
         same=["Reminder: the bonus still starts at 100 dollars for everyone.", "Nothing new: 100 dollars is the bonus threshold."],
         fmt=lambda a: f"amount {a['amount']} dollars, member: {'yes' if a['member'] else 'no'}",
         rule="An order gets a BONUS if the amount is 100 dollars or more, otherwise NONE."),
    dict(dom="library loans", noun="loan", attrs=[("days_late", list(range(0, 31))), ("student", [True, False])],
         old=lambda a: "FINE" if a["days_late"] > 7 else "OK",
         chg=["New library policy: students are never fined.", "Update: from today, no fines for students at all.",
              "Change in rules: students are exempt from fines."],
         new=lambda a: "OK" if a["student"] else ("FINE" if a["days_late"] > 7 else "OK"),
         same=["Reminder: the rule is the same for students and staff.", "No change: more than 7 days late means a FINE."],
         fmt=lambda a: f"{a['days_late']} days late, student: {'yes' if a['student'] else 'no'}",
         rule="A loan gets a FINE if it is more than 7 days late, otherwise OK."),
]


def t1(r, conf):
    R = r.choice(RULES)
    def case():
        return {k: r.choice(v) for k, v in R["attrs"]}
    n = r.choice([6, 8, 10, 12, 15])
    ex = "\n".join(f"{R['noun'].capitalize()} {i + 1}: {R['fmt'](a)} -> {R['old'](a)}" for i, a in enumerate(case() for _ in range(n)))
    while True:
        a = case()
        if conf == "yes" and R["old"](a) != R["new"](a):
            change = r.choice(R["chg"])
            if r.random() < 0.3:                        # cue crossing: a real change worded as a reminder
                change = r.choice(["Reminder of the change announced earlier: ", "Note, as mentioned before: "]) + change
            break
        if conf == "no":
            kind = r.random()
            if kind < 0.5 and R["old"](a) == R["new"](a):
                change = r.choice(R["chg"]); break      # rule changed, but not for this case
            if kind >= 0.5:
                change = r.choice(R["same"] + [f"Update: from now on we simply keep the same rule. {R['rule']}",
                                               f"Change log: none. {R['rule']}"]); break     # an 'update' that changes nothing
    intro = r.choice([f"We sort each {R['noun']} for {R['dom']}. Rule: {R['rule']}",
                      f"Here is our labeling rule for {R['dom']}. {R['rule']}",
                      f"Task: label each {R['noun']}. {R['rule']}"])
    q = r.choice([f"Now label this {R['noun']}: {R['fmt'](a)}", f"What is the label for: {R['fmt'](a)}?",
                  f"Next {R['noun']}: {R['fmt'](a)}. Label?"])
    msgs = sys_msgs(r) + [{"role": "user", "content": f"{intro}\n\nPast decisions:\n{ex}\n\n{change}\n\n{q}"}]
    if conf == "yes":
        w = why(r, [f"the past decisions follow the old rule and would give {R['old'](a)}, but the update gives {R['new'](a)} for this {R['noun']}.",
                    f"the worked examples point to {R['old'](a)}, while the changed rule now gives {R['new'](a)}.",
                    f"under the rule shown in the examples this is {R['old'](a)}; under the new rule it is {R['new'](a)}."])
        return msgs, target(True, "the past examples (old rule)", "the updated rule", w)
    w = why(r, [f"the old and the current rule give the same label ({R['new'](a)}) for this {R['noun']}.",
                f"the update does not change the outcome here; both point to {R['new'](a)}.",
                f"the examples and the rule in force agree on {R['new'](a)} for this case."])
    return msgs, target(False, "the past examples", "the rule in force", w)


# ---------- T2 instruction vs shown answers ----------
WORDS = ["apple", "river", "blue", "seven", "maple", "orbit", "candle", "pepper", "tiger", "violet"]
QS = ["What is the capital of France?", "How many legs does a spider have?", "What color is the sky on a clear day?",
      "What do bees make?", "Which planet is known as the Red Planet?", "What is 3 plus 4?", "What is frozen water called?",
      "How many days are in a week?", "What animal says moo?", "What is the largest ocean?", "What gas do plants take in?",
      "Who wrote Romeo and Juliet?", "What is the boiling point of water in Celsius?", "How many continents are there?"]


def t2(r, conf):
    said = r.choice(WORDS)
    shown = said if conf == "no" else r.choice([w for w in WORDS if w != said])
    k = r.choice([2, 3, 4, 6])
    qs = r.sample(QS, k + 1)
    instr = r.choice([f"For this whole conversation, answer every question with the single word '{said}'.",
                      f"Game rule: whatever I ask, reply only '{said}'.", f"Please answer each of my questions with just the word {said}."])
    msgs = sys_msgs(r) + [{"role": "user", "content": instr + " " + qs[0]}]
    for i in range(k):
        msgs += [{"role": "assistant", "content": shown.capitalize() + "."}, {"role": "user", "content": qs[i + 1]}]
    if conf == "yes":
        w = why(r, [f"the user asked for '{said}', but every earlier assistant reply says '{shown}', so copying the replies breaks the instruction.",
                    f"the instruction says '{said}' while the assistant's own past answers repeat '{shown}'.",
                    f"the pattern of previous replies ('{shown}') disagrees with what the user asked for ('{said}')."])
        return msgs, target(True, "the user's instruction", "the assistant's earlier replies", w)
    w = why(r, [f"the earlier replies already use '{said}', exactly as the user asked.",
                f"the instruction and the past replies agree on '{said}'.",
                f"the assistant has been answering '{said}' as instructed; nothing points elsewhere."])
    return msgs, target(False, "the user's instruction", "the assistant's earlier replies", w)


# ---------- T3 context claim vs world knowledge ----------
FACTS = [("a hexagon", "sides", 6), ("a spider", "legs", 8), ("a week", "days", 7), ("a year", "months", 12),
         ("a triangle", "corners", 3), ("an octopus", "arms", 8), ("a standard die", "faces", 6),
         ("a human hand", "fingers", 5), ("a chessboard", "squares per row", 8), ("a cube", "edges", 12)]


def t3(r, conf):
    subj, unit, M = r.choice(FACTS)
    v = M if conf == "no" else r.choice([x for x in range(2, 20) if abs(x - M) > 1])
    frame = r.choice([f"In my fantasy novel, {subj} has {v} {unit}.", f"For this puzzle, assume {subj} has {v} {unit}.",
                      f"My teacher's worksheet says {subj} has {v} {unit}.", f"According to the document I gave you, {subj} has {v} {unit}."])
    q = r.choice([f"How many {unit} does {subj} have?", f"Quick question: {subj} has how many {unit}?"])
    msgs = sys_msgs(r) + [{"role": "user", "content": f"{frame} {q}"}]
    if conf == "yes":
        w = why(r, [f"the text says {v} {unit}, but in reality {subj} has {M}.",
                    f"the user's stated premise ({v}) contradicts common knowledge ({M}).",
                    f"the claim given in the context ({v} {unit}) disagrees with the known fact ({M})."])
        return msgs, target(True, "the claim in the context", "general knowledge", w)
    w = why(r, [f"the context says {v} {unit}, which matches the real fact.",
                f"the premise and common knowledge agree on {M}.", f"nothing disagrees: {subj} has {M} {unit} both here and in reality."])
    return msgs, target(False, "the claim in the context", "general knowledge", w)


# ---------- T4 successive instructions ----------
FORMATS = [("in English", "in Spanish"), ("in one sentence", "as a bulleted list"), ("in all capital letters", "in lower case"),
           ("formally", "casually"), ("with an emoji at the end", "without any emoji"), ("in under 20 words", "in about 100 words"),
           ("as a haiku", "as a limerick"), ("signed 'Best, Ana'", "signed 'Cheers, Ana'")]
TASKS = ["write a thank-you note to my neighbor", "describe a rainy day", "explain what a battery does", "invite my team to lunch",
         "summarize why sleep matters", "describe my cat", "announce that the office is closed on Friday", "explain recycling"]


def t4(r, conf):
    a, b = r.choice(FORMATS)
    if r.random() < 0.5:
        a, b = b, a
    task = r.choice(TASKS)
    first = r.choice([f"Whenever I ask you to write something, write it {a}.", f"Rule for this chat: write everything {a}.",
                      f"Please make all your texts {a}."])
    if conf == "yes":
        later = r.choice([f"Actually, scratch that: from now on write {b} instead.", f"Correction: I want everything {b} now.",
                          f"Change of plan, please switch to writing {b}.", f"Please write {b}.", f"I'd like your texts {b} from here on.",
                          f"Reminder: from here on, everything {b}.", f"Just to repeat what I meant: write {b}.",
                          f"Update: please write {b} from now on.", f"Same task style as before, but write {b} now."])
    else:   # decoys: confirmations, and 'change' words about something that is not an instruction
        later = r.choice([f"Just to repeat: keep writing {a}.", f"Reminder: everything {a}, as before.",
                          "Also, thanks for following the rule so far.", "Change of plan: I'll read your text tomorrow instead of today.",
                          "Update: I'm on my phone now, so I may reply slowly.", "Correction: my last message had a typo, sorry.",
                          f"Actually, keep writing {a}, as before.", f"From now on, too, keep everything {a}."])
    msgs = sys_msgs(r) + [{"role": "user", "content": first}, {"role": "assistant", "content": "Understood."},
                          {"role": "user", "content": later}, {"role": "assistant", "content": "Okay."},
                          {"role": "user", "content": f"Now {task}."}]
    if conf == "yes":
        w = why(r, [f"the first instruction asks for writing {a}, the later one replaces it with {b}.",
                    f"an earlier rule ({a}) and a newer rule ({b}) call for different replies.",
                    f"the user changed the requirement from {a} to {b}."])
        return msgs, target(True, "the earlier instruction", "the later instruction", w)
    w = why(r, [f"the later message only repeats or confirms the first rule ({a}).", f"there is one requirement ({a}) and nothing replaces it.",
                f"no instruction was changed; both messages point to writing {a}."])
    return msgs, target(False, "the earlier instruction", "the later instruction", w)


# ---------- H1 (held out) stale personal value in a dialogue ----------
PREFS = [("favorite drink", ["latte", "green tea", "flat white", "hot chocolate", "espresso"]),
         ("home city", ["Lisbon", "Denver", "Osaka", "Nairobi", "Krakow"]),
         ("preferred seat", ["window seat", "aisle seat", "front row"]),
         ("dog's name", ["Biscuit", "Rex", "Luna", "Pepper"]),
         ("work schedule", ["morning shift", "night shift", "weekend shift"])]


def h1(r, conf):
    slot, vals = r.choice(PREFS)
    old = r.choice(vals); new = old if conf == "no" else r.choice([v for v in vals if v != old])
    nm = r.choice(NAMES)
    msgs = sys_msgs(r) + [{"role": "user", "content": f"Hi, I'm {nm}. My {slot} is {old}."},
                          {"role": "assistant", "content": f"Nice to meet you, {nm}! Noted: your {slot} is {old}."},
                          {"role": "user", "content": r.choice(["Can you help me plan my week?", "What's a good stretch for my back?",
                                                                 "Tell me a fun fact."])},
                          {"role": "assistant", "content": f"Sure! And of course I'll keep your {old} in mind."}]
    if conf == "yes":
        msgs.append({"role": "user", "content": r.choice([f"Oh, by the way, my {slot} is {new} now.", f"Update: it's {new} these days, not {old}.",
                                                          f"By the way, my {slot} is {new}.", f"Nothing else changed, but my {slot} is {new}.",
                                                          f"It's not the same anymore: my {slot} is {new}.",
                                                          f"I still remember telling you {old}, but it's {new} now.",
                                                          f"As before, I keep you posted: my {slot} is {new}."])})
    else:
        msgs.append({"role": "user", "content": r.choice([f"Oh, by the way, my {slot} is still {old}.", f"Yes, it's {old}, same as always.",
                                                          f"Update: my {slot} is {old}, as before.", f"Nothing changed, my {slot} is {old}.",
                                                          f"Just so you know, right now my {slot} is {old}.", f"By the way, my {slot} is {old}.",
                                                          f"I know I said it before, but my {slot} is {old}.", f"It's not different, my {slot} is {old}."])})
    msgs += [{"role": "assistant", "content": "Got it."}, {"role": "user", "content": f"So, what is my {slot}?"}]
    if conf == "yes":
        return msgs, target(True, f"the earlier value ({old})", f"the user's update ({new})",
                            f"the user first said {old} and the assistant repeated it, but later changed it to {new}.")
    return msgs, target(False, "the earlier value", "the user's later message", f"the user confirmed the same value, {old}.")


# ---------- H3 (held out) contradictory data in a word problem ----------
DATA = [("A crate holds {a} bottles.", "the crate, which holds {b} bottles,", "How many bottles are in 3 crates?"),
        ("The train leaves at {a}:00.", "the train, departing at {b}:00,", "How long until departure if it is 6:00 now?"),
        ("Each ticket costs {a} dollars.", "each ticket, priced at {b} dollars,", "What do 4 tickets cost?"),
        ("The garden is {a} meters long.", "the garden, {b} meters long,", "What is twice its length?"),
        ("Mia is {a} years old.", "Mia, who is {b},", "How old will she be in 5 years?")]


def h3(r, conf):
    s1, s2, q = r.choice(DATA)
    a = r.randint(7, 19); b = a if conf == "no" else r.choice([x for x in range(7, 20) if x != a])
    nm = r.choice(NAMES)
    body = f"{s1.format(a=a)} {nm} looked at {s2.format(b=b)} and wrote it down. {q}"
    msgs = sys_msgs(r) + [{"role": "user", "content": body}]
    if conf == "yes":
        return msgs, target(True, f"the first statement ({a})", f"the second statement ({b})", "the problem gives two different values for the same quantity.")
    return msgs, target(False, "the first statement", "the second statement", "both statements give the same value.")


FAM = dict(T1=t1, T2=t2, T3=t3, T4=t4, H1=h1, H3=h3)


def build(S):
    r = random.Random(S); out, seen = [], set()
    plan = [(f, "train", N_TRAIN) for f in ("T1", "T2", "T3", "T4")] + [(f, "dev", N_DEV) for f in ("T1", "T2", "T3", "T4")] + \
           [("H1", "test", N_HELD), ("H3", "test", N_HELD)]
    for f, split, n in plan:
        for i in range(n):
            conf = "yes" if i % 2 == 0 else "no"
            for _ in range(200):
                msgs, tgt = FAM[f](r, conf)
                key = json.dumps(msgs)
                if key not in seen:
                    break
            seen.add(key)
            out.append(dict(id=len(out), family=f, split=split, label=conf == "yes", messages=msgs, target=tgt))
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]); items = build(S)
    fn = os.path.join(HERE, "own_items.jsonl")
    with open(fn, "w") as fh:
        for x in items:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(dict(sorted(Counter((x["family"], x["split"], x["label"]) for x in items).items())))
