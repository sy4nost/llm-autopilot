import json, sys, collections, statistics
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("n", len(rows), "keys", list(rows[0].keys()))
for r in rows[:3]: print(json.dumps(r)[:700]); print('---')
nc=collections.Counter(len(r.get("classes",[])) for r in rows); print("n classes", nc)
ai=collections.Counter(r.get("answer_index") for r in rows); print("answer_index", ai)
# template families: first 40 chars
fam=collections.Counter(r["prompt"].split("\n")[0][:30] for r in rows); print(len(fam)); print(fam.most_common(15))
L=[len(r["prompt"]) for r in rows]; print("len", min(L), statistics.median(L), max(L))
