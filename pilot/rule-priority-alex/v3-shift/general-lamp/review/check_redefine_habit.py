"""Is the non-gold class in redefine-math always the 'habitual' (standard-math) answer? Reads data only."""
import json, sys, re, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
res=collections.Counter(); ex=[]
for r in rows:
    p=r["prompt"]; wrong=r["classes"][1-r["answer_index"]].strip(); gold=r["classes"][r["answer_index"]].strip()
    m=re.search(r"first digit of (\d+)([+÷])(\d+)\?", p)
    if "Redefine π" in p: hab="3"; t="pi"
    elif "Redefine e" in p: hab="2"; t="e"
    elif m:
        a,op,b=int(m[1]),m[2],int(m[3]); v=a+b if op=="+" else a/b
        hab=str(v)[0]; t="view"+op+("num" if "number as text" in p else "")
    else: t="?"; hab=None
    ok = hab==wrong
    res[(t,ok, gold==wrong)]+=1
    if not ok and len(ex)<6: ex.append((p[:120],r["classes"],r["answer_index"],hab))
for k,v in sorted(res.items()): print(k,v)
for e in ex: print(e)
