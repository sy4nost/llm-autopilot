"""Summary of the rule-switch test v2: reads trials.csv and layers.csv written by run_v2.py.

Pure Python (stdlib only). Items are the unit of inference: every confidence interval is a bootstrap over items
(seeded) or exact (Wilson for a rate, exact McNemar for a paired comparison of rates).
Terms (see run_v2.py): H = habit frame (practised by the habit history), S = switch frame (reverse mapping);
h = logp(habit answer) - logp(other answer), > 0 leans to the habit answer.

Usage: python analyze_v2.py <folder with trials.csv and layers.csv> [--boot 2000] [--boot-seed 1]
Prints the summary and writes summary.txt into the same folder (with run_info.txt, if present, on top).
"""
import argparse
import csv
import io
import math
import os
import random
from collections import Counter, defaultdict

GATE = 0.85
MIN_N = 2  # H1/H2 use N >= 2
LEVEL = 0.95
NAN = float("nan")
COND_ORDER = {"shared": 0, "habit": 1, "filler": 2}


def window(n_layers):
    """Pre-registered H2 window of logit-lens indices: round(0.50*L) .. round(0.67*L)-1, rounding half up.
    For L = 24 layers: 12..15."""
    lo = int(math.floor(0.50 * n_layers + 0.5))
    return lo, max(lo, int(math.floor(0.67 * n_layers + 0.5)) - 1)


def wilson(k, m, z=1.96):
    if m == 0:
        return NAN, NAN
    p, d = k / m, 1 + z * z / m
    c, h = (p + z * z / (2 * m)) / d, z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return c - h, c + h


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def mean_se(xs):
    xs = list(xs)
    if len(xs) < 2:
        return mean(xs), NAN
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def boot_ci(xs, n_boot, seed):
    """Percentile bootstrap CI of the mean, resampling items with replacement."""
    xs = list(xs)
    if len(xs) < 2:
        return NAN, NAN
    rng, m = random.Random(seed), len(xs)
    stats = sorted(sum(rng.choices(xs, k=m)) / m for _ in range(n_boot))
    cut = int((1 - LEVEL) / 2 * n_boot)
    return stats[cut], stats[n_boot - 1 - cut]


def ols_slope(xs, ys):
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def num(x, fmt="+.3f"):
    return "n/a" if x != x else format(x, fmt)


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def read_csv(path):
    """(rows, number of incomplete rows skipped). While run_v2.py is still writing, or after a run was killed, the
    file can end in the middle of a row: that last line (no line end) and any row with missing fields are skipped."""
    with open(path, newline="", encoding="utf-8") as f:
        text = f.read()
    skipped = 0
    if text and not text.endswith("\n"):
        text, skipped = text[:text.rfind("\n") + 1], 1
    rows = []
    for r in csv.DictReader(io.StringIO(text, newline="")):
        if None in r or None in r.values():
            skipped += 1
        else:
            rows.append(r)
    return rows, skipped


def paired(S, field, ns, keep=None):
    """Per N: {item: habit - filler} of `field` on S trials, for items with both conditions (and keep(item, N))."""
    per_n = {}
    for n in ns:
        h, f = S.get(("habit", n), {}), S.get(("filler", n), {})
        per_n[n] = {k: h[k][field] - f[k][field] for k in h if k in f and (keep is None or keep(k, n))}
    return per_n


def item_slopes(per_n, ns):
    """Per item: OLS slope of the paired difference against log2(N), for items present at every N. With every item at
    every N, the mean of these slopes equals the slope of the mean differences."""
    if len(ns) < 2:
        return []
    common = set.intersection(*(set(per_n[n]) for n in ns))
    xs = [math.log2(n) for n in ns]
    return [ols_slope(xs, [per_n[n][k] for n in ns]) for k in sorted(common)]


def analyze(folder, n_boot=2000, boot_seed=1, write=True, echo=True):
    """Build the summary; returns a dict of the key numbers (used by the tests)."""
    rows, cut_trials = read_csv(os.path.join(folder, "trials.csv"))
    if not rows:
        raise SystemExit(f"{folder}/trials.csv has no rows")
    for r in rows:
        r["n"], r["key"] = int(r["n"]), (r["seed"], int(r["item"]))
        for f in ("final_d", "final_h", "p_mass", "window_h"):
            r[f] = float(r[f])
    models = sorted({r["model"] for r in rows})
    if len(models) > 1:
        raise SystemExit(f"trials.csv mixes models {models}; analyze one model at a time")
    layer_path = os.path.join(folder, "layers.csv")
    layer_rows, cut_layers = read_csv(layer_path) if os.path.exists(layer_path) else ([], 0)
    n_layers = 1 + max(int(r["layer"]) for r in layer_rows) if layer_rows else None

    res = {"table": {}, "decision": {}, "window": window(n_layers) if n_layers else None}
    out = []
    say = out.append
    cells = defaultdict(list)
    for r in rows:
        cells[r["condition"], r["n"]].append(r)
    order = sorted(cells, key=lambda c: (COND_ORDER.get(c[0], 9), c[1]))
    S = defaultdict(dict)  # (condition, N) -> {item: row} for S trials
    info = {}
    for r in rows:
        info.setdefault(r["key"], r)
        if r["frame_role"] == "S":
            S[r["condition"], r["n"]][r["key"]] = r

    def ci(xs):
        return boot_ci(xs, n_boot, boot_seed)

    run_info = os.path.join(folder, "run_info.txt")
    if os.path.exists(run_info):
        with open(run_info, encoding="utf-8") as f:
            out.extend(f.read().rstrip("\n").split("\n"))
        say("")
    n_items = len(info)
    roles = Counter(r["frame_role"] for r in info.values())
    say(f"RULE-SWITCH TEST v2: SUMMARY   model={models[0]}   items={n_items} (H {roles['H']}, S {roles['S']})")
    win = res["window"]
    say(f"layers={n_layers or 'unknown (no layers.csv)'}; H2 window (pre-registered) = logit-lens indices"
        f" {f'{win[0]}..{win[1]}' if win else 'unknown'}; CIs 95%, bootstrap = {n_boot} resamples over items,"
        f" seed {boot_seed}")
    say("H = habit frame (the history practises it), S = switch frame (reverse mapping). h = logp(habit answer) -"
        " logp(other answer), in nats; > 0 leans to the habit (old) answer.")
    if cut_trials or cut_layers:
        say(f"WARNING: skipped incomplete rows (trials.csv {cut_trials}, layers.csv {cut_layers}): the run is still"
            " writing or was cut off. This summary covers only the rows written so far.")

    # 1. gate
    say("")
    say("1. GATE at N=0 (header + acquisition block, no history): both frames must be >= 85% correct.")
    g = cells.get(("shared", 0)) or cells.get(("habit", 0))
    gate_ok = None
    if not g:
        say("   no N=0 in this run: gate not checked.")
    else:
        parts = []
        accs = []
        for role, ok in (("H", "CORRECT"), ("S", "CORRECT_NEW_RULE")):
            sub = [r for r in g if r["frame_role"] == role]
            k = sum(r["error_type"] == ok for r in sub)
            lo, hi = wilson(k, len(sub))
            acc = k / len(sub) if sub else NAN
            accs.append(acc)
            parts.append(f"{role} frame {pct(acc)} [{pct(lo).strip()}-{pct(hi).strip()}] ({k}/{len(sub)})")
        gate_ok = all(a >= GATE for a in accs)
        say("   " + "   ".join(parts) + f"   -> {'PASS' if gate_ok else 'FAIL'}")
        if not gate_ok:
            say("   FAIL: the model does not apply both rules even without history. H1 and H2 below are NOT"
                " interpretable.")
    res["gate"] = None if gate_ok is None else ("PASS" if gate_ok else "FAIL")
    caveat = "" if gate_ok is not False else "  (gate FAILED: not interpretable)"

    # 2. per condition x N
    say("")
    say("2. BEHAVIOUR per condition and N (N=0 is one shared prompt).")
    say("   H acc = correct on H trials. S old = S trials answered by the habit rule, Wilson CI (50% = guessing).")
    say("   D = 2*S old - 1: +1 habit rule, 0 shape ignored, -1 switch rule (correct). S RIGHT = share of RIGHT on S"
        " (side bias).")
    say("   final_h S = mean h at the output on S trials, +-SE. p_mass = mean p(LEFT)+p(RIGHT) at the output.")
    say(f"   {'cond':>6} {'N':>3} {'H acc':>6} {'S old [95% CI]':>19} {'D':>6} {'S RIGHT':>8} {'final_h S':>14}"
        f" {'p_mass':>7}")
    for c in order:
        g = cells[c]
        hrows = [r for r in g if r["frame_role"] == "H"]
        srows = [r for r in g if r["frame_role"] == "S"]
        h_acc = mean(r["error_type"] == "CORRECT" for r in hrows)
        k = sum(r["error_type"] == "OLD_RULE_ERROR" for r in srows)
        old = k / len(srows) if srows else NAN
        lo, hi = wilson(k, len(srows))
        right = mean(r["choice"] == "RIGHT" for r in srows)
        fh, fse = mean_se(r["final_h"] for r in srows)
        pm = mean(r["p_mass"] for r in g)
        res["table"][c] = dict(h_acc=h_acc, s_old=old, D=2 * old - 1, s_right=right, final_h=fh, p_mass=pm)
        old_s = f"{pct(old)} [{pct(lo).strip()}-{pct(hi).strip()}]"
        say(f"   {c[0]:>6} {c[1]:>3} {pct(h_acc):>6} {old_s:>19} {2 * old - 1:>+6.2f} {pct(right):>8}"
            f" {fh:>+7.2f} +-{fse:<4.2f} {pm:>7.2f}")

    # H1 and H2 use N >= 2 (pre-registered); N = 0 is shared by both conditions, so it has no difference
    ns = sorted(n for n in {c[1] for c in S} if n >= MIN_N and ("habit", n) in S and ("filler", n) in S)
    span = f"N={','.join(map(str, ns))}" if ns else ""

    def per_n_table(per_n):
        say(f"     {'N':>3} {'items':>5} {'mean diff':>9}  95% CI")
        for n in ns:
            vals = list(per_n[n].values())
            lo, hi = ci(vals)
            say(f"     {n:>3} {len(vals):>5} {mean(vals):>+9.3f}  [{lo:+.3f}, {hi:+.3f}]")

    def slope_stats(per_n):
        sl = item_slopes(per_n, ns)
        return (mean(sl), *ci(sl), len(sl)) if sl else (NAN, NAN, NAN, 0)

    def group_line(label, per_n):
        m, lo, hi, k = slope_stats(per_n)
        parts = " ".join(f"{n}:{num(mean(per_n[n].values()), '+.2f')}" for n in ns)
        say(f"     {label:<19} {k:>4} items | by N {parts} | slope {num(m)} [{num(lo)}, {num(hi)}]")
        return m, lo, hi, k

    # 3. H1
    say("")
    say("3. H1 (primary, stated in advance): the habit history pushes S answers toward the habit rule, more so for")
    say("   longer N. Per item on S trials: final_h(habit) - final_h(filler) at the same N (> 0 = pulled to the old")
    say("   answer). H1 is supported if the slope of this difference against log2(N) has its CI above 0.")
    if len(ns) < 2:
        say(f"   needs at least two N >= {MIN_N} with both conditions: not tested.")
        res["H1"] = dict(supported=None)
    else:
        per_n = paired(S, "final_h", ns)
        per_n_table(per_n)
        m, lo, hi, k = slope_stats(per_n)
        sup = lo > 0
        say(f"   slope vs log2(N), {span}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {k} items"
            f"   -> H1 {'SUPPORTED' if sup else 'NOT SUPPORTED'}{caveat}")
        say("   Extra (not part of the verdict): the same by habit-answer side; opposite signs = side bias, not habit.")
        sides = {}
        for side in ("LEFT", "RIGHT"):
            sides[side] = group_line(f"habit ans {side}",
                                     paired(S, "final_h", ns, lambda key, n, s=side: info[key]["habit_answer"] == s))[0]
        res["H1"] = dict(slope=m, ci=(lo, hi), supported=sup, side_slopes=sides)
    if ns:
        say("   Secondary: OLD_RULE_ERROR habit vs filler, same items, exact McNemar.")
        say(f"     {'N':>3} {'habit':>6} {'filler':>6} {'only habit':>10} {'only filler':>11} {'p':>7}")
        for n in ns:
            h, f = S["habit", n], S["filler", n]
            keys = [k for k in h if k in f]
            oh = {k: h[k]["error_type"] == "OLD_RULE_ERROR" for k in keys}
            of = {k: f[k]["error_type"] == "OLD_RULE_ERROR" for k in keys}
            b, c = sum(oh[k] and not of[k] for k in keys), sum(of[k] and not oh[k] for k in keys)
            say(f"     {n:>3} {pct(mean(oh.values())):>6} {pct(mean(of.values())):>6} {b:>10} {c:>11}"
                f" {mcnemar(b, c):>7.3g}")
    hn = sorted(n for n in {c[1] for c in S} if n >= MIN_N and ("habit", n) in S)
    if len(hn) >= 2:
        a, z = S["habit", hn[0]], S["habit", hn[-1]]
        keys = [k for k in a if k in z]
        oa = {k: a[k]["error_type"] == "OLD_RULE_ERROR" for k in keys}
        oz = {k: z[k]["error_type"] == "OLD_RULE_ERROR" for k in keys}
        b, c = sum(oz[k] and not oa[k] for k in keys), sum(oa[k] and not oz[k] for k in keys)
        say(f"   Secondary: within habit, S old-rule N={hn[0]} {pct(mean(oa.values())).strip()} vs N={hn[-1]}"
            f" {pct(mean(oz.values())).strip()}; old-rule only at N={hn[-1]}: {b} items, only at N={hn[0]}: {c};"
            f" exact McNemar p = {mcnemar(b, c):.3g}")

    # 4. H2
    say("")
    say(f"4. H2 (primary, stated in advance): a hidden pull to the habit answer in the window layers"
        f" ({f'{win[0]}..{win[1]}' if win else '?'}).")
    say("   Per item on S trials: window_h(habit) - window_h(filler) at the same N, all S items. H2 is supported if")
    say("   the slope against log2(N) has its CI above 0 AND the slope CI is above 0 on each habit-answer side")
    say("   separately (LEFT items, RIGHT items), so that an effect on one side only cannot pose as a habit effect.")
    if len(ns) < 2:
        say(f"   needs at least two N >= {MIN_N} with both conditions: not tested.")
        res["H2"] = dict(supported=None)
    else:
        per_n = paired(S, "window_h", ns)
        per_n_table(per_n)
        m, lo, hi, k = slope_stats(per_n)
        say(f"   slope vs log2(N), {span}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {k} items")
        say("   Secondary (mean difference by N, slope with CI):")
        sides = {}
        for side in ("LEFT", "RIGHT"):
            sides[side] = group_line(f"habit ans {side}",
                                     paired(S, "window_h", ns, lambda key, n, s=side: info[key]["habit_answer"] == s))
        shapes = {}
        for shape in ("circle", "square"):
            shapes[shape] = group_line(f"shape {shape}",
                                       paired(S, "window_h", ns, lambda key, n, s=shape: info[key]["shape"] == s))[0]

        def both_correct(key, n, side):
            return (info[key]["habit_answer"] == side and S["habit", n][key]["error_type"] == "CORRECT_NEW_RULE"
                    and S["filler", n][key]["error_type"] == "CORRECT_NEW_RULE")
        both = {}
        for side in ("LEFT", "RIGHT"):
            both[side] = group_line(f"correct both, {side}",
                                    paired(S, "window_h", ns, lambda key, n, s=side: both_correct(key, n, s)))
        say("     (correct both = the S answer is correct in both conditions: at that N for 'by N', at every N for the")
        say("      slope; LEFT/RIGHT = habit answer. These items are chosen by the answer, so a model with a")
        say("      LEFT/RIGHT lean puts more of them on one side: count them as a hidden pull only if both agree.)")
        side_ok = all(v[1] > 0 for v in sides.values())
        sup = lo > 0 and side_ok
        why = [] if lo > 0 else ["slope CI not above 0"]
        why += [] if side_ok else ["slope CI not above 0 on both habit-answer sides"]
        why = "; ".join(why) or "slope CI above 0, overall and on each habit-answer side"
        say(f"   -> H2 {'SUPPORTED' if sup else 'NOT SUPPORTED'} ({why}){caveat}")
        res["H2"] = dict(slope=m, ci=(lo, hi), side_slopes={s: v[0] for s, v in sides.items()},
                         side_ci={s: v[1:3] for s, v in sides.items()}, shape_slopes=shapes,
                         correct_both={s: (v[0], v[3]) for s, v in both.items()}, supported=sup)

    # 5. decision layer
    say("")
    say("5. DECISION LAYER on correct H trials: the first logit-lens index where h exceeds half of its final value.")
    say("   mean curve = from the mean h per layer, averaged with equal weight over the two habit-answer sides (the")
    say("   lens has a LEFT/RIGHT lean per layer that cancels this way); trials L/R = correct H trials with habit")
    say("   answer LEFT / RIGHT. jump = rise of the mean curve at that index as a share of the final value (close to 1")
    say("   = the answer forms abruptly). per trial = the same index per trial after removing the cell's per-layer")
    say("   side lean: median [quartiles] and the most common index with its share.")
    say("   Indices count from 0: v1 (Qwen2.5-0.5B, 24 layers) found index 16 = the 17th layer.")
    if not layer_rows:
        say("   no layers.csv: skipped.")
    else:
        curves = defaultdict(dict)
        for r in layer_rows:
            if r["frame_role"] == "H" and r["error_type"] == "CORRECT":
                curves[r["condition"], int(r["n"]), (r["seed"], int(r["item"]))][int(r["layer"])] = float(r["h"])
        by_cell = defaultdict(lambda: {"LEFT": [], "RIGHT": []})  # (cond, N) -> habit answer -> curves
        for (cond, n, key), cur in curves.items():
            if len(cur) == n_layers and key in info:
                by_cell[cond, n][info[key]["habit_answer"]].append([cur[i] for i in range(n_layers)])
        say(f"   {'cond':>6} {'N':>3} {'trials L/R':>11} {'mean curve':>10} {'jump':>5}  per trial")
        pooled = Counter()
        for c in order:
            sides = by_cell.get(c, {"LEFT": [], "RIGHT": []})
            lr = f"{len(sides['LEFT'])}/{len(sides['RIGHT'])}"
            if not (sides["LEFT"] and sides["RIGHT"]):
                say(f"   {c[0]:>6} {c[1]:>3} {lr:>11}  (correct H trials on one habit-answer side or none: the side"
                    " lean cannot be removed, skipped)")
                continue
            sm = {s: [mean(x[i] for x in v) for i in range(n_layers)] for s, v in sides.items()}
            m = [(sm["LEFT"][i] + sm["RIGHT"][i]) / 2 for i in range(n_layers)]
            if m[-1] <= 0:
                say(f"   {c[0]:>6} {c[1]:>3} {lr:>11}  (mean final h <= 0)")
                continue
            d = next(i for i in range(n_layers) if m[i] > m[-1] / 2)
            jump = f"{(m[d] - m[d - 1]) / m[-1]:.2f}" if d else "-"
            res["decision"][c] = d
            # per trial: h minus the cell's side lean b = (mean LEFT - mean RIGHT) / 2 per layer, with the sign of
            # the trial's habit answer (h = signal + b on LEFT-habit trials, signal - b on RIGHT-habit trials)
            b = [(sm["LEFT"][i] - sm["RIGHT"][i]) / 2 for i in range(n_layers)]
            cur = ([[x[i] - b[i] for i in range(n_layers)] for x in sides["LEFT"]]
                   + [[x[i] + b[i] for i in range(n_layers)] for x in sides["RIGHT"]])
            per = sorted(next(i for i in range(n_layers) if x[i] > x[-1] / 2) for x in cur if x[-1] > 0)
            pooled.update(per)
            if per:
                q = [per[int(p * (len(per) - 1))] for p in (0.25, 0.5, 0.75)]
                mode, cnt = Counter(per).most_common(1)[0]
                pt = f"{q[1]} [{q[0]}-{q[2]}], most common {mode} ({cnt / len(per):.0%})"
            else:
                pt = "n/a"
            say(f"   {c[0]:>6} {c[1]:>3} {lr:>11} {d:>10} {jump:>5}  {pt}")
        cells_d = Counter(res["decision"].values())
        if cells_d:
            say("   distribution over the condition x N cells (mean curve): "
                + ", ".join(f"index {i}: {k}" for i, k in sorted(cells_d.items())) + f" (of {len(res['decision'])})")
        if pooled:
            tot = sum(pooled.values())
            say("   distribution over all correct H trials (per trial, side lean removed): "
                + ", ".join(f"{i}: {k / tot:.0%}" for i, k in sorted(pooled.items()) if k / tot >= 0.01)
                + " (indices under 1% not shown)")

    # 6. breakdowns
    say("")
    say("6. S OLD-RULE RATE BY FACTOR (each pair: first level / second level; 50% = guessing). A large gap inside a")
    say("   pair means that factor, not the habit, drives the answers. last hist same = the last history shape equals")
    say("   the test shape (habit, N > 0 only).")
    facs = [("habit ans L/R", "habit_answer", ("LEFT", "RIGHT")), ("shape c/s", "shape", ("circle", "square")),
            ("colour B/O", "habit_colour", ("BLUE", "ORANGE")), ("header H/S", "header_order", ("H_first", "S_first")),
            ("last acq H/S", "last_acq_role", ("H", "S")), ("last hist same/diff", "last_hist_shape_same", ("1", "0"))]
    say(f"   {'cond':>6} {'N':>3} " + " ".join(f"{name:>19}" for name, _, _ in facs))
    for c in order:
        srows = [r for r in cells[c] if r["frame_role"] == "S"]
        parts = []
        for _, field, levels in facs:
            vals = [mean(r["error_type"] == "OLD_RULE_ERROR" for r in srows if r[field] == lv) for lv in levels]
            parts.append("" if all(v != v for v in vals) else f"{pct(vals[0]).strip()} / {pct(vals[1]).strip()}")
        say(f"   {c[0]:>6} {c[1]:>3} " + " ".join(f"{p:>19}" for p in parts))

    text = "\n".join(out) + "\n"
    if echo:
        print(text, end="")
    if write:
        with open(os.path.join(folder, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        if echo:
            print(f"\nwritten: {os.path.join(folder, 'summary.txt')}")
    res["text"] = text
    return res


def main():
    ap = argparse.ArgumentParser(description="Summary of the rule-switch test v2 (see README.md)")
    ap.add_argument("folder", help="folder with trials.csv and layers.csv (the --out folder of run_v2.py)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--boot-seed", type=int, default=1)
    args = ap.parse_args()
    if args.boot < 1:
        ap.error("--boot must be at least 1")
    analyze(args.folder, n_boot=args.boot, boot_seed=args.boot_seed)


if __name__ == "__main__":
    main()
