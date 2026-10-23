"""v2 'fu / good' data: notice the change in any wording, and check whether it concerns THIS case.
New kinds on top of v1 (change, same, noisy):
  plain       - change stated in ordinary words (no 'NOTICE'), placed before or after the history   good=new  fu=old
  decoy       - a NOTICE about something else (another system / another person)                  good=old  fu=new
  decoy_plain - the same, in ordinary words                                                        good=old  fu=new
Training wording templates differ from the test/check wording (check.json, check2.json).
python3 -I make_fu2.py <out_dir> -> train2.json, check2.json"""
import json, random, sys
from make_fu import rule_item, fact_item, TRAIN_W, TEST_W, TRAIN_N, TEST_N, price_item, GOODS

R_PLAIN_TRAIN = ["Update from the team lead: Rule {k} should now be applied when {cond}.",
                 "Please note that Rule {k} was revised last week; it is now triggered when {cond}.",
                 "(Memo) Starting with this case, Rule {k} covers cases where {cond}.",
                 "The manager mentioned that Rule {k} has been rewritten: it now fires when {cond}."]
R_DECOY_TRAIN = ["NOTICE: in the archive department's separate system (not this one), Rule {k} now applies when {cond}.",
                 "NOTICE: the training manual for next year will change Rule {k} to apply when {cond}. Not in force yet.",
                 "NOTICE: a proposal to make Rule {k} apply when {cond} was rejected; the rules stay as they are."]
R_DECOY_PLAIN_TRAIN = ["Someone suggested that Rule {k} should apply when {cond}, but the suggestion was not accepted.",
                       "In the partner company, Rule {k} is used when {cond}; our own rules are unchanged."]
F_PLAIN_TRAIN = ["Email from facilities: {who} has been moved to {thing} {new} starting today.",
                 "By the way, {who} switched to {thing} {new} this morning.",
                 "(Change) {who}'s {thing} is now {new}."]
F_DECOY_TRAIN = ["NOTICE: from today {other} is assigned {thing} {new}; the old one was given away.",
                 "NOTICE: {thing} {new} has been repainted. Nothing else changed."]
F_DECOY_PLAIN_TRAIN = ["By the way, {other} switched to {thing} {new} this morning.",
                       "{who} asked to move to {thing} {new}, but the request was declined."]


def split_rule(it):
    p = it["prompt"]
    a = p.index("NOTICE: Rule ")
    b = p.index("\n\n", a)
    note = p[a:b]
    k = note[len("NOTICE: Rule "):].split(" ")[0]
    cond = note.split("It now applies when ", 1)[1].split(" (instead of ")[0]
    return p[:a], p[b + 2:], k, cond


def place(rng, head, tail, line):
    """put the line right before the case (default) or before the past decisions"""
    if rng.random() < 0.5 or "Past decisions" not in head:
        return head + line + "\n\n" + tail
    i = head.index("Past decisions")
    return head[:i] + line + "\n\n" + head[i:] + tail


def rule_v2(rng, W, kind, plain_t, decoy_t, decoy_plain_t):
    it = rule_item(rng, W, "change")
    head, tail, k, cond = split_rule(it)
    if kind == "plain":
        it["prompt"] = place(rng, head, tail, rng.choice(plain_t).format(k=k, cond=cond))
        it["fu"] = it["old"]
    else:
        t = decoy_t if kind == "decoy" else decoy_plain_t
        it["prompt"] = place(rng, head, tail, rng.choice(t).format(k=k, cond=cond))
        it["key"], it["fu"] = it["old"], it["key"]   # good = keep the experience; fu = apply the irrelevant change
    it["kind"] = kind
    return it


def fact_v2(rng, names, kind, plain_t, decoy_t, decoy_plain_t):
    it = fact_item(rng, names, "change")
    p = it["prompt"]
    a = p.index("NOTICE: from today ")
    b = p.index("\n\n", a)
    who = p[a + len("NOTICE: from today "):].split(" ")[0]
    thing = p.split("which ", 1)[1].split(" will ")[0]
    new = it["key"]
    logs = [l.split(": ", 1)[1].split(" ")[0] for l in p.split("\n") if l.startswith("Log ")]
    other = rng.choice(sorted(set(logs) - {who}))
    if kind == "plain":
        line = rng.choice(plain_t).format(who=who, thing=thing, new=new)
        it["fu"] = it["old"]
    else:
        line = rng.choice(decoy_t if kind == "decoy" else decoy_plain_t).format(who=who, other=other, thing=thing, new=new)
        it["key"], it["fu"] = it["old"], new
    it["prompt"] = p[:a] + line + "\n\n" + p[b + 2:]
    it["kind"] = kind
    return it


# ----- held-out check wording (never used in training)
R_PLAIN_TEST = ["From today the office applies Rule {k} only when {cond}.", "Heads-up: Rule {k} now goes by {cond}."]
R_DECOY_TEST = ["NOTICE: the sister office has its own rulebook; there, Rule {k} applies when {cond}.",
                "Draft for discussion only: Rule {k} could one day apply when {cond}."]
F_PLAIN_TEST = ["{who} told us that their {thing} is {new} from now on."]
F_DECOY_TEST = ["NOTICE: {other} now uses {thing} {new}.", "{who} wanted {thing} {new}, but it was already taken, so nothing changed."]


def price_v2(rng, kind):
    """never-trained family. plain2: change in other words, placed after the history; decoy_plain: other service, plain words"""
    if kind == "plain":
        it = price_item(rng, "change")
        p = it["prompt"]
        a = p.index("\nNOTICE: the ")
        b = p.index("\n", a + 1)
        srv, rest = p[a + len("\nNOTICE: the "):b].split(" price is now ")
        it["prompt"] = p[:a] + f"\nThe {srv} rate changed today: it is {rest.rstrip('.')} now." + p[b:]
    else:
        it = price_item(rng, "same")
        srv = it["prompt"].split("Past ", 1)[1].split(" orders")[0]
        o_srv, o_unit, _ = rng.choice([g for g in GOODS if g[0] != srv])
        x = rng.randint(3, 12)
        it["prompt"] = f"By the way, {o_srv} now costs ${x} per {o_unit}.\n\n" + it["prompt"]
        q = int(it["prompt"].split("New order: ")[1].split(" ")[2])
        wrong = str(q * x) if str(q * x) != it["key"] else str(q * x + 1)
        it["cands"] = [it["key"], wrong]
    it["kind"] = kind
    return it


def build():
    rng = random.Random(20261018)
    tr = []
    tr += [rule_item(rng, TRAIN_W, "change") for _ in range(350)]
    tr += [rule_item(rng, TRAIN_W, "same") for _ in range(250)]
    tr += [rule_item(rng, TRAIN_W, "noisy") for _ in range(200)]
    tr += [rule_v2(rng, TRAIN_W, "plain", R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN) for _ in range(350)]
    tr += [rule_v2(rng, TRAIN_W, "decoy", R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN) for _ in range(250)]
    tr += [rule_v2(rng, TRAIN_W, "decoy_plain", R_PLAIN_TRAIN, R_DECOY_TRAIN, R_DECOY_PLAIN_TRAIN) for _ in range(150)]
    tr += [fact_item(rng, TRAIN_N, "change") for _ in range(150)]
    tr += [fact_item(rng, TRAIN_N, "same") for _ in range(100)]
    tr += [fact_v2(rng, TRAIN_N, "plain", F_PLAIN_TRAIN, F_DECOY_TRAIN, F_DECOY_PLAIN_TRAIN) for _ in range(150)]
    tr += [fact_v2(rng, TRAIN_N, "decoy", F_PLAIN_TRAIN, F_DECOY_TRAIN, F_DECOY_PLAIN_TRAIN) for _ in range(100)]
    tr += [fact_v2(rng, TRAIN_N, "decoy_plain", F_PLAIN_TRAIN, F_DECOY_TRAIN, F_DECOY_PLAIN_TRAIN) for _ in range(50)]
    rng.shuffle(tr)
    ck = []
    ck += [rule_v2(rng, TEST_W, "plain", R_PLAIN_TEST, R_DECOY_TEST, R_DECOY_TEST) for _ in range(60)]
    ck += [rule_v2(rng, TEST_W, "decoy", R_PLAIN_TEST, R_DECOY_TEST, R_DECOY_TEST) for _ in range(60)]
    ck += [fact_v2(rng, TEST_N, "plain", F_PLAIN_TEST, F_DECOY_TEST, F_DECOY_TEST) for _ in range(40)]
    ck += [fact_v2(rng, TEST_N, "decoy", F_PLAIN_TEST, F_DECOY_TEST, F_DECOY_TEST) for _ in range(40)]
    ck += [price_v2(rng, "plain") for _ in range(60)]
    ck += [price_v2(rng, "decoy_plain") for _ in range(60)]
    for x in ck:
        x["kind"] = x["kind"] + "2"
    return tr, ck


if __name__ == "__main__":
    tr, ck = build()
    json.dump(tr, open(sys.argv[1] + "/train2.json", "w"), indent=1)
    json.dump(ck, open(sys.argv[1] + "/check2.json", "w"), indent=1)
    from collections import Counter
    print("train2", len(tr), Counter((x["fam"], x["kind"]) for x in tr))
    print("check2", len(ck), Counter((x["fam"], x["kind"]) for x in ck))
    bad = [x for x in tr + ck if x["key"] == x.get("fu")]
    print("pairs with key==fu:", len(bad))
