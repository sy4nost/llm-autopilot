"""Check whether the trained model learned an attitude or just the word 'NOTICE'.
  rule/price 'plain'  - the change is said in plain words, no 'NOTICE' header         key = new
  rule/price 'decoy'  - a NOTICE is present but is about something else             key = old (copying is right)
python3 -I make_check.py <out_dir> -> check.json"""
import json, random, sys
sys.path.insert(0, "")
from make_fu import rule_item, price_item, TEST_W, GOODS

rng = random.Random(7)
out = []
for _ in range(60):  # rule plain
    it = rule_item(rng, TEST_W, "change")
    a = it["prompt"].index("NOTICE: Rule ")
    b = it["prompt"].index("\n\n", a)
    note = it["prompt"][a:b]
    k = note[len("NOTICE: Rule "):].split(" ")[0]
    rest = note.split("It now applies when ", 1)[1].split(" (instead of ")[0]
    plain = f"The staff were told this morning that from now on Rule {k} is used when {rest}."
    it["prompt"] = it["prompt"][:a] + plain + it["prompt"][b:]
    it["kind"] = "plain"
    out.append(it)
for _ in range(60):  # rule decoy: irrelevant notice, nothing changed
    it = rule_item(rng, TEST_W, "same")
    i = it["prompt"].index("CASE:")
    it["prompt"] = it["prompt"][:i] + "NOTICE: the storage room has moved to building B. Nothing else has changed.\n\n" + it["prompt"][i:]
    it["kind"] = "decoy"
    out.append(it)
for _ in range(60):  # price plain
    it = price_item(rng, "top")
    first = it["prompt"].split("\n\n", 1)
    srv_rate = first[0][len("NOTICE: the "):]          # "<srv> price is now $X per <unit>."
    srv, rest = srv_rate.split(" price is now ")
    it["prompt"] = f"Since this morning the {srv} costs {rest.rstrip('.')}.\n\n" + first[1]
    it["kind"] = "plain"
    out.append(it)
for _ in range(60):  # price decoy: notice about a different service, this price unchanged
    it = price_item(rng, "same")
    srv = it["prompt"].split("Past ", 1)[1].split(" orders")[0]
    o_srv, o_unit, _ = rng.choice([g for g in GOODS if g[0] != srv])
    x = rng.randint(3, 12)
    it["prompt"] = f"NOTICE: the {o_srv} price is now ${x} per {o_unit}.\n\n" + it["prompt"]
    q = int(it["prompt"].split("New order: ")[1].split(" ")[2])
    wrong = str(q * x)
    if wrong == it["key"]:
        wrong = str(q * x + 1)
    it["cands"] = [it["key"], wrong]
    it["kind"] = "decoy"
    out.append(it)
json.dump(out, open(sys.argv[1] + "/check.json", "w"), indent=1)
print(len(out))
for k in range(0, 240, 60):
    print("=====", out[k]["fam"], out[k]["kind"], out[k]["key"], out[k]["old"], out[k]["cands"])
    print(out[k]["prompt"][-420:])
