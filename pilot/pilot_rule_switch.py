"""Mini-pilot: rule-switch test for a small open LLM (roadmap step 1).

Scheme from Impulse Lab Experiment 2: a usual rule (BLUE frame: circle -> LEFT, square -> RIGHT) and a switched rule
(ORANGE frame: circle -> RIGHT, square -> LEFT). Each test prompt is:
  rule header -> N solved history trials -> a short balanced rule-learning block -> one test trial.
Two history conditions, same length and same shapes:
  habit: all N history trials are BLUE (the usual rule is automatized);
  mixed: the N history trials alternate BLUE / ORANGE (control: same prompt length, no automatization).
H1 is tested as "habit minus mixed" growing with N, so prompt length alone cannot produce it.

Every prompt is independent, so the test-frame ratio is not a manipulation here (unlike the human 80/20); tests are 50/50
BLUE/ORANGE and balanced by shape. Items are paired across N and conditions: item i always has the same test cell,
learn block and history shapes (the last N of one long history).

We read the model's log-probabilities of " LEFT" and " RIGHT" for the answer (no sampling), and with the logit lens the
same comparison at every layer.
  pull at a layer = logprob(wrong) - logprob(correct); > 0 means that layer leans to the wrong answer.
  On ORANGE trials the wrong answer is the habitual (old-rule) one.
  final_pull  = pull at the output (continuous version of OLD_RULE_ERROR).
  hidden_pull = max pull over the upper half of the layers, final layer excluded (early-layer logit-lens readouts are
                close to a constant per layer and would turn the statistic into a LEFT/RIGHT bias).
"Hidden impulse": an ORANGE trial answered correctly whose hidden_pull exceeds the threshold. As in Impulse Lab
(90th percentile of the participant's own deviation on clean trials, with a floor), the threshold is the 90th percentile
of hidden_pull on correct BLUE trials of the same condition, N and correct answer, floored at 0. By construction about
10% of correct BLUE trials exceed it; that BLUE rate is printed next to the ORANGE rate as the null reference, and H2 is
judged by ORANGE minus BLUE.

Error types (Impulse Lab README): CORRECT / OTHER_ERROR on BLUE, CORRECT_NEW_RULE / OLD_RULE_ERROR on ORANGE.

Usage:
  pip install torch transformers
  python pilot_rule_switch.py                       # Qwen/Qwen2.5-0.5B-Instruct, CPU is fine
  python pilot_rule_switch.py --model <hf id> --trials 100 --n 0 10 30 60 --learn 4
Outputs: results/trials.csv (one row per trial), results/layers.csv (logit-lens pull per layer), summary on screen.
"""
import argparse
import csv
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SHAPES = ["circle", "square"]
FRAMES = ["BLUE", "ORANGE"]
RULE = {"BLUE": {"circle": "LEFT", "square": "RIGHT"}, "ORANGE": {"circle": "RIGHT", "square": "LEFT"}}
HEADER = (
    "You are playing a reaction game. Each trial shows a shape inside a colored frame.\n"
    "BLUE frame (usual rule): circle -> LEFT, square -> RIGHT.\n"
    "ORANGE frame (switched rule): circle -> RIGHT, square -> LEFT.\n"
    "Answer each trial with one word: LEFT or RIGHT.\n\n"
)


def solved(frame, shape):
    return f"Frame: {frame}. Shape: {shape}. Answer: {RULE[frame][shape]}\n"


def make_items(args):
    """Paired items: test cell, learn block and history shapes are fixed per item and reused for every N/condition."""
    cells = [(f, s) for f in FRAMES for s in SHAPES for _ in range(args.trials // 4)]
    random.Random(args.seed).shuffle(cells)
    n_max = max(args.n)
    items = []
    for i, (frame, shape) in enumerate(cells):
        rng = random.Random(f"{args.seed}-{i}")
        history = [rng.choice(SHAPES) for _ in range(n_max)]
        # mixed control: the same shapes; frames alternate, counted from the end so every suffix stays balanced
        mixed_frames = [FRAMES[(n_max - k) % 2] for k in range(n_max)]
        learn = [(f, s) for f in FRAMES for s in SHAPES for _ in range(args.learn // 4)]
        rng.shuffle(learn)
        # counterbalance the frame of the last learn example across items (it is the closest copy cue to the test)
        want_last = FRAMES[i % 2]
        if learn and learn[-1][0] != want_last:
            j = max(k for k, (f, _) in enumerate(learn) if f == want_last)
            learn[j], learn[-1] = learn[-1], learn[j]
        items.append(dict(item=i, frame=frame, shape=shape, history=history, mixed_frames=mixed_frames,
                          learn=learn, last_learn_frame=learn[-1][0] if learn else ""))
    return items


def build_prompt(item, n, condition):
    lines = [HEADER]
    hist = item["history"][len(item["history"]) - n:] if n else []
    frames = item["mixed_frames"][len(item["mixed_frames"]) - n:] if n else []
    for k, s in enumerate(hist):
        lines.append(solved("BLUE" if condition == "habit" else frames[k], s))
    for f, s in item["learn"]:
        lines.append(solved(f, s))
    lines.append(f"Frame: {item['frame']}. Shape: {item['shape']}. Answer:")
    return "".join(lines)


def answer_token_ids(tok):
    # first token of " LEFT" / " RIGHT" (leading space: the answer follows "Answer:"); single tokens in Qwen2.5
    ids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in ("LEFT", "RIGHT")}
    assert ids["LEFT"] != ids["RIGHT"], "LEFT and RIGHT share a first token; pick other answer words"
    return ids


def final_norm(model):
    # Qwen/Llama/Mistral: model.model.norm; GPT-2 style: model.transformer.ln_f
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    return getattr(inner, "norm", None) or getattr(inner, "ln_f")


@torch.no_grad()
def run_trial(model, tok, ids, prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True)
    norm, head = final_norm(model), model.get_output_embeddings()
    # logit lens: last-position hidden state after each layer -> final norm -> unembedding.
    # hidden_states[0] is the embeddings; the last entry is already normalized by the model, so the last layer is
    # read straight from the logits (applying the norm twice would distort it).
    per_layer = []
    for h in out.hidden_states[1:-1]:
        lp = torch.log_softmax(head(norm(h[:, -1, :]))[0].float(), dim=-1)
        per_layer.append((lp[ids["LEFT"]].item(), lp[ids["RIGHT"]].item()))
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    final = (lp[ids["LEFT"]].item(), lp[ids["RIGHT"]].item())
    per_layer.append(final)
    return final, per_layer


def classify(frame, shape, choice):
    if frame == "BLUE":
        return "CORRECT" if choice == RULE["BLUE"][shape] else "OTHER_ERROR"
    # with two options, the only wrong ORANGE answer is the usual-rule one
    return "CORRECT_NEW_RULE" if choice == RULE["ORANGE"][shape] else "OLD_RULE_ERROR"


def p90(values):
    v = sorted(values)
    return v[int(0.9 * (len(v) - 1))] if v else float("inf")


def wilson(k, m, z=1.96):
    if m == 0:
        return float("nan"), float("nan")
    p, d = k / m, 1 + z * z / m
    c, h = (p + z * z / (2 * m)) / d, z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return c - h, c + h


def mean_se(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else float("nan")
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--trials", type=int, default=100, help="test items, a multiple of 4 (50/50 BLUE/ORANGE x shape)")
    ap.add_argument("--n", type=int, nargs="+", default=[0, 10, 30, 60], help="history lengths")
    ap.add_argument("--learn", type=int, default=4, help="rule-learning examples, a multiple of 4 (0 = none)")
    ap.add_argument("--no-control", action="store_true", help="skip the mixed-history control condition")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    assert args.trials >= 4 and args.trials % 4 == 0, "--trials must be a multiple of 4"
    assert args.learn % 4 == 0, "--learn must be a multiple of 4"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # .float() instead of a dtype argument: works on old and new transformers versions alike
    model = AutoModelForCausalLM.from_pretrained(args.model).float().to(device).eval()
    ids = answer_token_ids(tok)
    os.makedirs(args.out, exist_ok=True)

    items = make_items(args)
    conditions = ["habit"] if args.no_control else ["habit", "mixed"]
    rows, layer_rows = [], []
    for cond in conditions:
        for n in args.n:
            if cond == "mixed" and n == 0:
                continue  # identical to habit at N=0
            for it in items:
                (lpL, lpR), per_layer = run_trial(model, tok, ids, build_prompt(it, n, cond))
                choice = "LEFT" if lpL > lpR else "RIGHT"
                correct = RULE[it["frame"]][it["shape"]]
                pulls = [(r - l) if correct == "LEFT" else (l - r) for l, r in per_layer]
                n_layers = len(pulls)
                upper = pulls[n_layers // 2:n_layers - 1] or pulls[-1:]
                err = classify(it["frame"], it["shape"], choice)
                rows.append(dict(condition=cond, n=n, item=it["item"], frame=it["frame"], shape=it["shape"],
                                 correct=correct, choice=choice, error_type=err,
                                 last_learn_frame=it["last_learn_frame"], side_bias=round(lpL - lpR, 4),
                                 final_pull=round(pulls[-1], 4), hidden_pull=round(max(upper), 4)))
                for layer, p in enumerate(pulls):
                    layer_rows.append(dict(condition=cond, n=n, item=it["item"], frame=it["frame"],
                                           error_type=err, layer=layer, pull=round(p, 4)))
            print(f"{cond} N={n}: done", flush=True)

    # thresholds per (condition, N, correct answer) from correct BLUE trials, floored at 0
    thr = {}
    for cond in conditions:
        for n in args.n:
            for ct in ("LEFT", "RIGHT"):
                v = [r["hidden_pull"] for r in rows if (r["condition"], r["n"], r["correct"]) == (cond, n, ct)
                     and r["error_type"] == "CORRECT"]
                thr[cond, n, ct] = max(0.0, p90(v))
    for r in rows:
        above = r["hidden_pull"] > thr[r["condition"], r["n"], r["correct"]]
        r["above_threshold"] = int(above)
        r["hidden_impulse"] = int(above and r["error_type"] == "CORRECT_NEW_RULE")

    for name, data in (("trials.csv", rows), ("layers.csv", layer_rows)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"\nmodel={args.model}  device={device}  items={len(items)} (ORANGE {len(items) // 2})  chance=50%")
    print("OLD_RULE_ERROR = wrong ORANGE answers (mean of circle and square rates, 95% CI);"
          " pull = mean final pull on ORANGE (nats, +SE);")
    print("hidden = ORANGE correct above threshold vs BLUE correct above threshold (null ~10%);"
          " bias = mean logp(L)-logp(R)")
    print(f"{'cond':>6} {'N':>3} {'BLUE acc':>8} {'OLD_RULE_ERROR':>22} {'pull':>14} {'hidden ORANGE':>14}"
          f" {'null BLUE':>10} {'bias':>6}")
    for cond in conditions:
        for n in args.n:
            g = [r for r in rows if r["condition"] == cond and r["n"] == n]
            if not g:
                continue
            b = [r for r in g if r["frame"] == "BLUE"]
            o = [r for r in g if r["frame"] == "ORANGE"]
            k = sum(r["error_type"] == "OLD_RULE_ERROR" for r in o)
            lo, hi = wilson(k, len(o))
            by_shape = [sum(r["error_type"] == "OLD_RULE_ERROR" for r in o if r["shape"] == s)
                        / max(1, sum(r["shape"] == s for r in o)) for s in SHAPES]
            pm, pse = mean_se([r["final_pull"] for r in o])
            ok_o = [r for r in o if r["error_type"] == "CORRECT_NEW_RULE"]
            ok_b = [r for r in b if r["error_type"] == "CORRECT"]
            hid = f"{sum(r['hidden_impulse'] for r in ok_o)}/{len(ok_o)}"
            null = f"{sum(r['above_threshold'] for r in ok_b)}/{len(ok_b)}"
            acc_b = sum(r["error_type"] == "CORRECT" for r in b) / len(b)
            ore = f"{sum(by_shape) / 2:.0%} [{lo:.0%}-{hi:.0%}]"
            print(f"{cond:>6} {n:>3} {acc_b:>8.0%} {ore:>22} {pm:>+8.2f}±{pse:<5.2f} {hid:>14} {null:>10}"
                  f" {sum(r['side_bias'] for r in g) / len(g):>+6.2f}")
    print(f"\nfiles: {args.out}/trials.csv, {args.out}/layers.csv")


if __name__ == "__main__":
    main()
