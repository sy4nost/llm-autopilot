"""Expected positives and the AUC of a 'surface-only' predictor that knows only the prompt cell (arm x probe type),
using habit-error rates from earlier runs (match/score_out.txt, native/score_native_out.txt, newtasks2/score_new2_out.txt,
public/score_public_out.txt). Not a model run: a back-of-envelope of how much within-family AUC is available from surface alone."""
def auc(cells):  # cells: list of (n, rate); score = rate (cell mean)
    pos = [(n*r, r) for n, r in cells]; neg = [(n*(1-r), r) for n, r in cells]
    P = sum(p for p, _ in pos); N = sum(q for q, _ in neg); s = 0.0
    for p, rp in pos:
        for q, rq in neg:
            s += p*q*(1.0 if rp > rq else 0.5 if rp == rq else 0.0)
    return P, N, s/(P*N)
fam = {
 # C-rule: 30 sets; per arm D1+D3 (60 prompts) and D2+D4 (60); nochange: habit error impossible (key = old)
 "C-rule": [(60,.93),(60,.10),(60,.38),(60,.22),(60,.12),(60,.33),(120,0.0)],
 # P-native change_0: D1+D3 81%, D2+D4 0% (100% right)
 "P-native": [(60,.81),(60,0.0)],
 # parcels (20 sets): A60 T, E_nocopy T, E_copy T, N60 T, S in all 4 arms
 "C-parcels": [(40,.80),(40,.88),(40,0.0),(40,0.0),(160,0.0)],
 "C-bonus": [(40,.90),(40,.90),(40,.05),(40,0.0),(160,0.0)],
}
for k, v in fam.items():
    P, N, a = auc(v); print(f"{k:10s} n={sum(n for n,_ in v):4d} expected positives={P:5.0f} negatives={N:5.0f}  cell-only AUC={a:.3f}")
# C-families restricted to trap situations only (key != old, model either kept old or not)
print("--- restricted to trap situations (the target the public families use)")
for k, v in {"C-rule": [(60,.93),(60,.10),(60,.38),(60,.22),(60,.12),(60,.33)], "P-native": [(60,.81),(60,0.0)],
             "C-parcels": [(40,.80),(40,.88),(40,0.0)], "C-bonus": [(40,.90),(40,.90),(40,.05)]}.items():
    P, N, a = auc(v); print(f"{k:10s} expected positives={P:5.0f} negatives={N:5.0f}  cell-only AUC={a:.3f}")
print("--- public, errors expected at earlier accuracy (n=200 sample in public/)")
for k, acc, ntot in (("redefine-math", .69, 900), ("quote-repetition", .86, None), ("NeQA", .615, None)):
    for n in (200, 300, 500, 900):
        print(f"{k:17s} n={n:4d} expected errors={n*(1-acc):5.0f}")
