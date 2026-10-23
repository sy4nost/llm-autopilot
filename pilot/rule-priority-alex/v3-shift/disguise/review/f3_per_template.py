"""Per-template accuracy / mass of the normal model in the general-lamp main run (F3). python3 -I f3_per_template.py <meta_main.jsonl>"""
import json, sys, math
from collections import defaultdict
st = defaultdict(lambda: [0, 0, 0, 0])
nat = defaultdict(lambda: [0, 0, 0])
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("family") == "P-redefine":
        sc = r["scores"]; m = sum(math.exp(s) for s in sc)
        pick = max(range(len(sc)), key=lambda j: sc[j])
        a = st[r["cell"]]; a[0] += 1; a[1] += m >= .5
        if m >= .5:
            a[2] += pick == r["gold"]
        a[3] += pick == r["gold"]
    if r.get("family") == "P-native":
        sc = r["scores"]; c = r["cands"]; pick = c[max(range(3), key=lambda j: sc[j])]
        cell = r["cell"]; b = nat[cell]; b[0] += 1; b[1] += pick == r["key"]; b[2] += pick == r["old"]
print("P-redefine (F3) cell: n, kept(mass>=.5), correct among kept, correct all")
for k in sorted(st): print(" ", k, st[k])
print("P-native change_0 cell: n, correct(new rule), old-rule answer")
for k in sorted(nat): print(" ", k, nat[k])
