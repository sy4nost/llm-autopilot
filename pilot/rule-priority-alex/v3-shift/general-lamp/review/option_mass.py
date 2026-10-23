"""Probability mass the model puts on the answer options (summed log-prob, as in public/modal_public.py), per task."""
import json, sys, math, statistics, collections
by=collections.defaultdict(list)
for l in open(sys.argv[1]):
    if not l.strip(): continue
    r=json.loads(l)
    if "normal" not in r or not isinstance(r["normal"], list): continue
    sc=r["normal"]; m=max(sc); lse=m+math.log(sum(math.exp(s-m) for s in sc))
    by[r["task"]].append((lse, int(max(range(len(sc)),key=lambda k:sc[k])!=r["gold"])))
for t,v in by.items():
    L=sorted(x for x,_ in v); q=lambda p:L[int(p*(len(L)-1))]
    print(f"{t:26s} n={len(v)} log P(any option): median {statistics.median(L):7.2f}  p10 {q(.1):7.2f}  p90 {q(.9):7.2f}  share>log(0.5): {sum(x>math.log(.5) for x in L)/len(L):.2f}  errors {sum(e for _,e in v)}")
