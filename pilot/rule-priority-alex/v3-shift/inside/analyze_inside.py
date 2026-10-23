"""Analyze inside_<model>.jsonl.  python3 -I analyze_inside.py <file.jsonl>
margin = logp(new-rule label) - logp(old-rule label) at the answer position (first label token); > 0 = new wins.
Groups: D1+D3 (old HOLD -> new not HOLD; the habit errors), D2+D4 (new adds HOLD; no habit errors).
Prints per history N: final margin, % old wins; logit-lens margin by layer (every 4th); attention shares by layer
(averaged over the late half); knockouts (margin when the answer cannot attend to examples / to the update line),
with paired bootstrap over sets.
"""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
rng = random.Random(1)
SEGS = ["rules", "examples", "update", "test", "other"]
bad = [r for r in rows if len(set(r["label_tok"])) < 3]
print(f"{len(rows)} prompts; label first-token collisions: {len(bad)}; max mask_check {max(r['mask_check'] for r in rows)}")
rows = [r for r in rows if len(set(r["label_tok"])) == 3]


def idx(r, lab):
    return r["labels"].index(lab)


def margin(r, vec):
    return vec[idx(r, r["key"])] - vec[idx(r, r["old"])]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def boot(d):
    b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
    return f"{mean(d):+.2f} [{b[100]:+.2f}, {b[3899]:+.2f}]"


def per_set(rs, f):
    by = {}
    for r in rs:
        by.setdefault(r["set"], []).append(f(r))
    return {s: mean(v) for s, v in by.items()}


L = len(rows[0]["lens"])
for grp, pids in (("D1+D3 (habit errors)", ("D1", "D3")), ("D2+D4 (no habit errors)", ("D2", "D4"))):
    print(f"\n===== {grp} =====")
    for n in ("0", "20", "60"):
        rs = [r for r in rows if r["arm"] == f"change_{n}" and r["probe"] in pids]
        fm = [margin(r, r["final"]) for r in rs]
        print(f"\nN={n}: final margin new-old {mean(fm):+.2f} nats; old label wins in {mean([m < 0 for m in fm]):.0%}")
        lens = [mean([margin(r, r["lens"][l]) for r in rs]) for l in range(L)]
        print("  logit lens margin by layer: " + " ".join(f"{l}:{lens[l]:+.1f}" for l in range(0, L, 4)) + f" {L - 1}:{lens[-1]:+.1f}")
        late = range(L // 2, L)
        att = [mean([mean([r["attn"][l][k] for l in late if r["attn"][l]]) for r in rs]) for k in range(len(SEGS))]
        print("  attention from answer, late layers: " + ", ".join(f"{s} {a:.1%}" for s, a in zip(SEGS, att)))
        for ko in ("K_ex", "K_up"):
            rk = [r for r in rs if r["knockout"][ko]]
            if not rk:
                continue
            km = [margin(r, r["knockout"][ko]) for r in rk]
            d = per_set(rk, lambda r: margin(r, r["knockout"][ko]) - margin(r, r["final"]))
            print(f"  knockout {ko}: margin {mean(km):+.2f}, old wins {mean([m < 0 for m in km]):.0%}; "
                  f"change vs normal (per set) {boot(list(d.values()))}")
# attention to the update line vs N, all D probes, by layer thirds
print("\n===== attention to the update line and to the examples (all D probes), by depth =====")
thirds = [range(0, L // 3), range(L // 3, 2 * L // 3), range(2 * L // 3, L)]
for n in ("0", "20", "60"):
    rs = [r for r in rows if r["arm"] == f"change_{n}"]
    cells = []
    for name, rg in zip(("early", "middle", "late"), thirds):
        up = mean([mean([r["attn"][l][2] for l in rg if r["attn"][l]]) for r in rs])
        ex = mean([mean([r["attn"][l][1] for l in rg if r["attn"][l]]) for r in rs])
        cells.append(f"{name}: update {up:.1%} examples {ex:.1%}")
    print(f"N={n}: " + " | ".join(cells))
