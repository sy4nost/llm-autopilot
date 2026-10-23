"""Rule-switch test v2 for a small open LLM (roadmap step 1, second version).

Two rule frames with symmetric roles. H ("habit") is the frame the history practises, S ("switch") has the reverse
mapping. A third, neutral frame (GRAY, any shape -> SKIP) is used for the control history.
Six factors are crossed exactly, so every combination is equally frequent (64 cells, --trials a multiple of 64):
  habit colour (BLUE/ORANGE), habit mapping (circle->LEFT or circle->RIGHT), header order (H line first or S line
  first), test frame role (H/S), test shape (circle/square), frame role of the last acquisition example (H/S).
Because the habit mapping is balanced, the old-rule answer is LEFT as often as RIGHT, and a LEFT/RIGHT bias of the
model can no longer pose as a habit effect.

Prompt = header -> acquisition block -> history (N lines) -> test line.
  header:      game line, the two rule lines (in the item's header order), the GRAY line, "Answer each trial ...";
  acquisition: --learn solved examples, balanced over H/S x circle/square, shuffled per item;
  history:     habit  = N solved H-frame trials (random shapes per item);
               filler = N lines "Frame: GRAY. Shape: triangle|star. Answer: SKIP" (same line format, no mapping);
  test:        "Frame: <COLOUR>. Shape: <shape>. Answer:".
The history for N is the first N lines of one per-item sequence, so prompts for growing N are prefixes of each other.
N=0 is the same prompt in both conditions; it is run once and labelled "shared".
Items are paired: test trial, acquisition block and history shapes are fixed per item and reused for every N/condition.

Readout (no sampling): log-probabilities of the first tokens of " LEFT" and " RIGHT" at the last position, and the same
at every layer with the logit lens (final norm + unembedding on hidden_states[1:-1]; the last layer is read from the
logits, because hidden_states[-1] is already normalized).
  h        = logp(habit answer) - logp(other answer); > 0 leans to the habit (H) answer. Side-free by design.
  final_h  = h at the output; final_d = logp(LEFT) - logp(RIGHT) at the output; p_mass = p(LEFT) + p(RIGHT).
  window_h = mean h over the pre-registered window of logit-lens indices round(0.50*L) .. round(0.67*L)-1
             (L = number of transformer layers; 12..15 for 24 layers).

Speed: for each item and condition the longest prefix (header + acquisition + history[:max N]) is run once with the KV
cache; then, for N in descending order, the cache is cropped back to the prefix length of that N and only the test line
is run. If the tokenization of a prompt is not exactly "prefix tokens + test tokens", that N gets a full forward pass.
  --no-cache       always run the full prompt (slow, for comparison)
  --check-cache K  for the first K items also run the full prompt and compare; stops if any value differs by > 1e-3

Usage:
  pip install torch transformers
  python run_v2.py                                   # Qwen/Qwen2.5-0.5B-Instruct, 512 items, CPU is fine
  python run_v2.py --model Qwen/Qwen2.5-1.5B-Instruct
  python run_v2.py --dry-run                         # only build and check the items, print prompts of 2 items
  python analyze_v2.py <out folder>                  # summary again, without the model
Outputs in --out (default results_v2/<model>-seed<seed>): trials.csv, layers.csv, run_info.txt, summary.txt.
"""
import argparse
import csv
import importlib.util
import inspect
import itertools
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

SHAPES = ["circle", "square"]
COLOURS = ["BLUE", "ORANGE"]
ROLES = ["H", "S"]
MAPS = {"circle_left": {"circle": "LEFT", "square": "RIGHT"}, "circle_right": {"circle": "RIGHT", "square": "LEFT"}}
HEADER_ORDERS = ["H_first", "S_first"]
FILLER_COLOUR, FILLER_SHAPES, FILLER_ANSWER = "GRAY", ["triangle", "star"], "SKIP"
GAME_LINE = "You are playing a reaction game. Each trial shows a shape inside a colored frame.\n"
# the six crossed factors, two levels each -> 64 cells
FACTORS = [("habit_colour", COLOURS), ("habit_map", list(MAPS)), ("header_order", HEADER_ORDERS),
           ("frame_role", ROLES), ("shape", SHAPES), ("last_acq_role", ROLES)]
N_CELLS = 2 ** len(FACTORS)
CHECK_LIMIT = 1e-3
TRIAL_FIELDS = ["seed", "model", "condition", "n", "item", "habit_colour", "habit_map", "header_order", "frame_role",
                "colour", "shape", "last_acq_role", "last_hist_shape_same", "correct", "habit_answer", "choice",
                "error_type", "final_d", "final_h", "p_mass", "window_h"]
LAYER_FIELDS = ["seed", "condition", "n", "item", "frame_role", "shape", "error_type", "layer", "h"]


def load_analyzer():
    # load analyze_v2.py from this script's folder by its path (also works under "python -I");
    # no __pycache__ folder is written next to the scripts
    sys.dont_write_bytecode = True
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_v2.py")
    spec = importlib.util.spec_from_file_location("analyze_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def other(x, pair):
    return pair[1] if x == pair[0] else pair[0]


def solved(colour, shape, answer):
    return f"Frame: {colour}. Shape: {shape}. Answer: {answer}\n"


def make_items(trials, learn, n_max, seed):
    """Paired items: the 64 cells repeated trials/64 times, shuffled; everything random is seeded per item."""
    cells = [c for c in itertools.product(*(levels for _, levels in FACTORS)) for _ in range(trials // N_CELLS)]
    random.Random(seed).shuffle(cells)
    items = []
    for i, cell in enumerate(cells):
        it = dict(zip([name for name, _ in FACTORS], cell))
        it["item"] = i
        colour = {"H": it["habit_colour"], "S": other(it["habit_colour"], COLOURS)}
        rule = {"H": MAPS[it["habit_map"]], "S": MAPS[other(it["habit_map"], list(MAPS))]}
        it["colour"] = colour[it["frame_role"]]
        it["correct"] = rule[it["frame_role"]][it["shape"]]
        it["habit_answer"] = rule["H"][it["shape"]]
        order = ROLES if it["header_order"] == "H_first" else ROLES[::-1]
        it["header"] = (GAME_LINE
                        + "".join(f"{colour[r]} frame: circle -> {rule[r]['circle']}, square -> {rule[r]['square']}.\n"
                                  for r in order)
                        + f"{FILLER_COLOUR} frame: any shape -> {FILLER_ANSWER}.\n"
                        + "Answer each trial with one word.\n\n")
        acq = [(r, s) for r in ROLES for s in SHAPES for _ in range(learn // 4)]
        random.Random(f"{seed}-{i}-acq").shuffle(acq)
        if acq[-1][0] != it["last_acq_role"]:  # factor 6: the role of the example closest to the history/test
            j = max(k for k, (r, _) in enumerate(acq) if r == it["last_acq_role"])
            acq[j], acq[-1] = acq[-1], acq[j]
        it["acq"] = acq
        it["acq_text"] = "".join(solved(colour[r], s, rule[r][s]) for r, s in acq)
        # separate generators, so the first N lines do not depend on the largest N of the grid
        hist_rng, fill_rng = random.Random(f"{seed}-{i}-habit"), random.Random(f"{seed}-{i}-filler")
        it["hist_shapes"] = [hist_rng.choice(SHAPES) for _ in range(n_max)]
        it["history"] = {
            "habit": [solved(colour["H"], s, rule["H"][s]) for s in it["hist_shapes"]],
            "filler": [solved(FILLER_COLOUR, fill_rng.choice(FILLER_SHAPES), FILLER_ANSWER) for _ in range(n_max)]}
        it["test"] = f"Frame: {it['colour']}. Shape: {it['shape']}. Answer:"
        items.append(it)
    return items


def prefix_text(it, cond, n):
    return it["header"] + it["acq_text"] + "".join(it["history"][cond][:n])


def prompt_text(it, cond, n):
    return prefix_text(it, cond, n) + it["test"]


def conditions_for(ns):
    """(condition, Ns) pairs to run per item: habit carries N=0 (reported as shared), filler only N>0."""
    out = [("habit", sorted(ns, reverse=True))]
    if any(n > 0 for n in ns):
        out.append(("filler", sorted((n for n in ns if n > 0), reverse=True)))
    return out


def check_items(items, trials, learn, ns):
    """Balance and pairing checks; raises AssertionError on any failure."""
    counts = Counter(tuple(it[name] for name, _ in FACTORS) for it in items)
    assert len(counts) == N_CELLS and set(counts.values()) == {trials // N_CELLS}, f"unbalanced cells: {counts}"
    for it in items:
        acq = Counter(it["acq"])
        assert len(acq) == 4 and set(acq.values()) == {learn // 4}, f"item {it['item']}: unbalanced acquisition"
        assert it["acq"][-1][0] == it["last_acq_role"], f"item {it['item']}: wrong last acquisition role"
        assert it["correct"] == it["habit_answer"] if it["frame_role"] == "H" else it["correct"] != it["habit_answer"]
        start = it["header"] + it["acq_text"]
        prompts = {}
        for cond, cond_ns in conditions_for(ns):
            prev = None
            for n in sorted(cond_ns):
                p = prompt_text(it, cond, n)
                assert p.startswith(start) and p.endswith("\n" + it["test"]), f"item {it['item']}: broken prompt"
                pre = prefix_text(it, cond, n)
                assert prev is None or pre.startswith(prev), f"item {it['item']}: prefixes do not nest"
                prev = pre
                prompts[cond, n] = p
        if 0 in ns and ("filler", 0) not in prompts:
            assert prompt_text(it, "filler", 0) == prompts["habit", 0], "N=0 prompt differs between conditions"


def answer_token_ids(tok):
    # first token of " LEFT" / " RIGHT" (leading space: the answer follows "Answer:"); single tokens in Qwen2.5
    ids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in ("LEFT", "RIGHT")}
    assert ids["LEFT"] != ids["RIGHT"], "LEFT and RIGHT share a first token; pick other answer words"
    return ids


def final_norm(model):
    # Qwen/Llama/Mistral: model.model.norm; GPT-2 style: model.transformer.ln_f
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    return getattr(inner, "norm", None) or getattr(inner, "ln_f")


def encode(tok, text, special=True):
    return tok(text, add_special_tokens=special)["input_ids"]


def crop_cache(cache, length):
    """Cut the KV cache back to its first `length` tokens. crop() gets a negative number (tokens to remove): this form
    works on transformers 4.x and 5.x (5.x refuses a positive target length)."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)
    if cache.get_seq_length() != length:
        raise RuntimeError(f"cache crop failed: length {cache.get_seq_length()}, wanted {length}")


class Runner:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device
        self.ids = answer_token_ids(tok)
        self.norm, self.head = final_norm(model), model.get_output_embeddings()
        params = inspect.signature(model.forward).parameters
        # only the last position's logits are needed; skips the unembedding of the whole prompt
        self.kw = {"logits_to_keep": 1} if "logits_to_keep" in params else {}
        self.n_layers = model.config.num_hidden_layers
        self.fallbacks = 0
        self.prompts = 0
        self.cache_ok = True  # set to False if this transformers version gives a cache without crop()

    def tensor(self, ids):
        return torch.tensor([ids], device=self.device)

    @torch.no_grad()
    def readout(self, out):
        """[(logp LEFT, logp RIGHT)] for every layer at the last position; the last entry is the model output."""
        lr = [self.ids["LEFT"], self.ids["RIGHT"]]
        # embeddings + one entry per layer; otherwise the layer numbering below would be wrong
        if len(out.hidden_states) != self.n_layers + 1:
            raise RuntimeError(f"model returned {len(out.hidden_states)} hidden states, expected {self.n_layers + 1}")
        rows = []
        if len(out.hidden_states) > 2:
            mid = torch.stack([h[0, -1, :] for h in out.hidden_states[1:-1]])
            rows = torch.log_softmax(self.head(self.norm(mid)).float(), dim=-1)[:, lr].tolist()
        rows.append(torch.log_softmax(out.logits[0, -1].float(), dim=-1)[lr].tolist())
        return [tuple(r) for r in rows]

    @torch.no_grad()
    def full(self, it, cond, n):
        out = self.model(input_ids=self.tensor(encode(self.tok, prompt_text(it, cond, n))), use_cache=False,
                         output_hidden_states=True, **self.kw)
        return self.readout(out)

    @torch.no_grad()
    def run(self, it, cond, ns, use_cache):
        """{N: readout} for one item and condition; ns in descending order."""
        res = {}
        test_ids = encode(self.tok, it["test"], special=False)
        use_cache = use_cache and self.cache_ok
        if use_cache:
            long_ids = encode(self.tok, prefix_text(it, cond, max(ns)))
            cache = self.model(input_ids=self.tensor(long_ids), use_cache=True, **self.kw).past_key_values
            if not (hasattr(cache, "crop") and hasattr(cache, "get_seq_length")):
                print("WARNING: this transformers version returns a KV cache that cannot be cropped;"
                      " every prompt is run in full (slower, same results)", flush=True)
                self.cache_ok = use_cache = False
        for n in ns:
            self.prompts += 1
            if use_cache:
                pre_ids = encode(self.tok, prefix_text(it, cond, n))
                full_ids = encode(self.tok, prompt_text(it, cond, n))
                if full_ids == pre_ids + test_ids and long_ids[:len(pre_ids)] == pre_ids:
                    crop_cache(cache, len(pre_ids))
                    out = self.model(input_ids=self.tensor(test_ids), past_key_values=cache, use_cache=True,
                                     output_hidden_states=True, **self.kw)
                    cache = out.past_key_values
                    res[n] = self.readout(out)
                    continue
                self.fallbacks += 1
            res[n] = self.full(it, cond, n)
        return res


def trial_rows(it, cond, n, lp, win, seed, model_name):
    lpL, lpR = lp[-1]
    choice = "LEFT" if lpL > lpR else "RIGHT"
    if it["frame_role"] == "H":
        err = "CORRECT" if choice == it["correct"] else "OTHER_ERROR"
    else:  # with two options, the only wrong S answer is the habit one
        err = "CORRECT_NEW_RULE" if choice == it["correct"] else "OLD_RULE_ERROR"
    hs = [(l - r) if it["habit_answer"] == "LEFT" else (r - l) for l, r in lp]
    label = "shared" if n == 0 else cond
    same = int(it["hist_shapes"][n - 1] == it["shape"]) if cond == "habit" and n > 0 else ""
    row = dict(seed=seed, model=model_name, condition=label, n=n, item=it["item"], habit_colour=it["habit_colour"],
               habit_map=it["habit_map"], header_order=it["header_order"], frame_role=it["frame_role"],
               colour=it["colour"], shape=it["shape"], last_acq_role=it["last_acq_role"], last_hist_shape_same=same,
               correct=it["correct"], habit_answer=it["habit_answer"], choice=choice, error_type=err,
               final_d=round(lpL - lpR, 4), final_h=round(hs[-1], 4), p_mass=round(math.exp(lpL) + math.exp(lpR), 4),
               window_h=round(sum(hs[win[0]:win[1] + 1]) / (win[1] - win[0] + 1), 4))
    layers = [dict(seed=seed, condition=label, n=n, item=it["item"], frame_role=it["frame_role"], shape=it["shape"],
                   error_type=err, layer=k, h=round(h, 4)) for k, h in enumerate(hs)]
    return row, layers


def show_prompts(items, ns, k=2):
    n_show = sorted(n for n in ns if n > 0)[:1] or [0]
    for it in items[:k]:
        for cond, cond_ns in conditions_for(ns):
            if n_show[0] in cond_ns:
                print(f"--- item {it['item']}, {cond}, N={n_show[0]} "
                      f"(habit {it['habit_colour']} {it['habit_map']}, test {it['frame_role']} {it['shape']},"
                      f" correct {it['correct']}) ---")
                print(prompt_text(it, cond, n_show[0]))
        print()


def fmt_time(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 60 else f"{s:.0f} s"


def main():
    ap = argparse.ArgumentParser(description="Rule-switch test v2 (see README.md)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--trials", type=int, default=512, help="test items, a multiple of 64 (all 64 cells equally)")
    ap.add_argument("--n", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32, 64], help="history lengths")
    ap.add_argument("--learn", type=int, default=16, help="acquisition examples, a multiple of 4, at least 4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="output folder (default results_v2/<model>-seed<seed>)")
    ap.add_argument("--no-cache", action="store_true", help="run every prompt in full (slow)")
    ap.add_argument("--check-cache", type=int, default=0, metavar="K",
                    help="for the first K items compare cached and full runs; stop if they differ by > 1e-3")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples in the summary")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="weights precision; float32 on CPU, bfloat16 on a GPU for 7B models")
    ap.add_argument("--dry-run", action="store_true", help="build and check the items, print prompts of 2 items, exit")
    args = ap.parse_args()
    ns = sorted(set(args.n))
    assert args.trials >= N_CELLS and args.trials % N_CELLS == 0, "--trials must be a multiple of 64"
    assert args.learn >= 4 and args.learn % 4 == 0, "--learn must be a multiple of 4 (at least 4)"
    assert ns[0] >= 0, "--n values must be >= 0"
    assert args.boot >= 1, "--boot must be at least 1"
    analyzer = load_analyzer()
    # last part of the model id or local path ("Qwen/Qwen2.5-0.5B-Instruct" -> "Qwen2.5-0.5B-Instruct"), also for
    # Windows paths like D:\models\qwen
    model_tag = re.sub(r"[^A-Za-z0-9._-]", "_", re.split(r"[\\/]", args.model.rstrip("/\\"))[-1]) or "model"
    out_dir = args.out or os.path.join("results_v2", f"{model_tag}-seed{args.seed}")

    items = make_items(args.trials, args.learn, max(ns), args.seed)
    check_items(items, args.trials, args.learn, ns)
    print(f"items: {len(items)}, all 64 cells x {args.trials // N_CELLS}, pairing checked")
    if args.dry_run:
        show_prompts(items, ns)
        return

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # load, then cast with .to(dtype) instead of a dtype argument: works on old and new transformers versions alike
    model = AutoModelForCausalLM.from_pretrained(args.model).to(getattr(torch, args.dtype)).to(device).eval()
    if args.dtype != "float32":
        global CHECK_LIMIT
        CHECK_LIMIT = 5e-2  # half precision: cached and full runs take different kernels, small differences are normal
    runner = Runner(model, tok, device)
    n_layers = model.config.num_hidden_layers
    win = analyzer.window(n_layers)
    plan = conditions_for(ns)
    use_cache = not args.no_cache
    os.makedirs(out_dir, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M")
    print(f"model={args.model} device={device} layers={n_layers} H2 window = logit-lens indices {win[0]}..{win[1]}"
          f"  cache={'on' if use_cache else 'off'}  out={out_dir}")

    f_trials = open(os.path.join(out_dir, "trials.csv"), "w", newline="", encoding="utf-8")
    f_layers = open(os.path.join(out_dir, "layers.csv"), "w", newline="", encoding="utf-8")
    w_trials, w_layers = csv.DictWriter(f_trials, TRIAL_FIELDS), csv.DictWriter(f_layers, LAYER_FIELDS)
    w_trials.writeheader()
    w_layers.writeheader()
    check_max, check_n = 0.0, 0
    t_start = time.perf_counter()
    for k, it in enumerate(items):
        for cond, cond_ns in plan:
            res = runner.run(it, cond, cond_ns, use_cache)
            if k < args.check_cache:
                for n, lp in res.items():
                    ref = runner.full(it, cond, n)
                    assert len(ref) == len(lp) == n_layers
                    check_max = max([check_max] + [abs(a - b) for x, y in zip(lp, ref) for a, b in zip(x, y)])
                    check_n += 1
            for n in cond_ns:
                assert len(res[n]) == n_layers, "readout length differs from the number of layers"
                row, layers = trial_rows(it, cond, n, res[n], win, args.seed, args.model)
                w_trials.writerow(row)
                w_layers.writerows(layers)
        f_trials.flush()
        f_layers.flush()
        if args.check_cache and k + 1 == min(args.check_cache, len(items)):
            verdict = "OK" if check_max <= CHECK_LIMIT else "FAILED"
            print(f"check-cache: {k + 1} items, {check_n} prompts, max abs diff cached vs full = {check_max:.2e}"
                  f" (limit {CHECK_LIMIT:g}): {verdict}", flush=True)
            if check_max > CHECK_LIMIT:
                f_trials.close()
                f_layers.close()
                raise SystemExit("check-cache FAILED: cached and full runs differ; rerun with --no-cache")
        done = time.perf_counter() - t_start
        if k < 2 or (k + 1) % 10 == 0 or k + 1 == len(items):
            per = done / (k + 1)
            print(f"item {k + 1}/{len(items)}: {per:.1f} s per item, left about {fmt_time(per * (len(items) - k - 1))}",
                  flush=True)
    f_trials.close()
    f_layers.close()
    total = time.perf_counter() - t_start
    n_ic = len(items) * len(plan)
    info = [
        f"run_v2.py, started {started}, finished {time.strftime('%Y-%m-%d %H:%M')}",
        f"model={args.model} device={device} dtype={args.dtype} python={platform.python_version()} torch={torch.__version__}"
        f" transformers={transformers.__version__} threads={torch.get_num_threads()}",
        f"items={len(items)} learn={args.learn} N={' '.join(map(str, ns))} seed={args.seed}"
        f" cache={'on' if use_cache and runner.cache_ok else 'off'} layers={n_layers} window={win[0]}..{win[1]}",
        f"time: {fmt_time(total)} total, {total / n_ic:.2f} s per item-condition ({n_ic} item-conditions,"
        f" {runner.prompts} prompts)",
        f"cache fallbacks (full forward because tokenization was not prefix + test): {runner.fallbacks}"
        f" of {runner.prompts if use_cache and runner.cache_ok else 0}",
        f"check-cache: {'not run' if not args.check_cache else f'{check_n} prompts, max abs diff {check_max:.2e} OK'}",
    ]
    with open(os.path.join(out_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print()
    analyzer.analyze(out_dir, n_boot=args.boot)  # prints run_info.txt and the summary, writes summary.txt


if __name__ == "__main__":
    main()
