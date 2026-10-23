import json, sys, collections, re
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def tmpl(p):
    p=re.sub(r"\d+","#",p); return p[:45]
g=collections.defaultdict(list)
for r in rows: g[tmpl(r["prompt"])].append(r)
# coarse
c2=collections.defaultdict(list)
for k,v in g.items(): c2[k[:25]].extend(v)
for k,v in sorted(c2.items(), key=lambda x:-len(x[1])):
    ai=collections.Counter(r["answer_index"] for r in v)
    print(len(v), dict(ai), repr(k)); 
    for r in v[:2]: print("   ", r["prompt"][:160].replace("\n"," | "), r["classes"], r["answer_index"])
