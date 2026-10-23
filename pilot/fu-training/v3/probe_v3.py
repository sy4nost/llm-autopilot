import json, sys
from collections import Counter
r = [json.loads(l) for l in open(sys.argv[1])]
D = {(x["model"], x["task"], x["i"]): x for x in r}
for t in dict.fromkeys(x["task"] for x in r):
    ids = [x["i"] for x in r if x["task"] == t and x["model"] == "base"]
    b = [D[("base", t, i)] for i in ids]; v = [D[("v3", t, i)] for i in ids]
    if "scores" in b[0]:
        pick = lambda x: max(range(len(x["scores"])), key=lambda k: x["scores"][k])
        gold = Counter(x["gold"] for x in b)
        pb = Counter(pick(x) for x in b); pv = Counter(pick(x) for x in v)
        flip = sum(pick(x) != pick(y) for x, y in zip(b, v))
        to_right = sum(pick(x) != pick(y) and pick(y) == y["gold"] for x, y in zip(b, v))
        # accuracy per gold label
        acc = {g: (round(100*sum(pick(x)==g for x in b if x["gold"]==g)/gold[g]), round(100*sum(pick(y)==g for y in v if y["gold"]==g)/gold[g])) for g in sorted(gold)}
        print(f"{t:26} gold={dict(gold)} pick base={dict(pb)} v3={dict(pv)} flips={flip} (to right {to_right}, to wrong {flip-to_right}) acc_by_gold base/v3={acc}")
    else:
        none_b = sum(x["pred"] in (None, "") for x in b); none_v = sum(y["pred"] in (None, "") for y in v)
        both = Counter((x["pred"] == x["gold"], y["pred"] == y["gold"]) for x, y in zip(b, v))
        lb = sum(len(x["text"]) for x in b)/len(b); lv = sum(len(y["text"]) for y in v)/len(v)
        print(f"{t:26} no-answer base={none_b} v3={none_v}  (base ok, v3 ok) counts={dict(both)}")
