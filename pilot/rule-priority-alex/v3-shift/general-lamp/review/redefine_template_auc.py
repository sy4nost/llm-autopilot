"""Per-template habit-error rate of 14B on the 200 redefine-math items of public/ (normal arm), and the AUC of a
template-only predictor, plus prompt-length and margin AUCs. Reads data files only."""
import json, sys, random, collections
data=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pub=[json.loads(l) for l in open(sys.argv[2]) if l.strip()]
pub=[r for r in pub if r["task"]=="redefine-math"]
idx=list(range(len(data))); random.Random(0).shuffle(idx); rows=[data[i] for i in sorted(idx[:len(pub)])]
def tmpl(p):
    if "Redefine π as" in p: return "pi:"+("mod" if "mod" in p else "+" if "+" in p.split("?")[0].split("as")[1] else "num")
    if "Redefine e" in p: return "e+"
    if "number as text" in p: return "view-num"
    if "÷" in p: return "view÷:"+("instead" if "instead" in p else "short")
    return "view+:"+("random" if "random" in p else "digit")
cells=collections.defaultdict(lambda:[0,0]); ys=[]; feats=[]
for r,o in zip(rows,pub):
    assert r["answer_index"]==o["gold"], "alignment"
    sc=o["normal"]; err=int(max(range(2),key=lambda k:sc[k])!=o["gold"])
    t=tmpl(r["prompt"]); cells[t][0]+=1; cells[t][1]+=err; ys.append(err)
    feats.append((t,len(r["prompt"]),abs(sc[0]-sc[1])))
for t,(n,e) in sorted(cells.items()): print(f"{t:18s} n={n:3d} errors={e:3d} rate={e/n:.2f}")
def auc(s,y):
    P=[a for a,b in zip(s,y) if b]; N=[a for a,b in zip(s,y) if not b]
    return sum((p>q)+.5*(p==q) for p in P for q in N)/(len(P)*len(N))
rate={t:e/n for t,(n,e) in cells.items()}
print("errors",sum(ys),"of",len(ys))
print("template-only AUC (in-sample cell means):", round(auc([rate[f[0]] for f in feats],ys),3))
print("prompt-length AUC:", round(auc([f[1] for f in feats],ys),3))
print("margin AUC (small margin = fires):", round(auc([-f[2] for f in feats],ys),3))
