"""Disguise / shake test (PLAN.md v2) on Qwen2.5-14B-Instruct. Every answer option is scored by its own full forward pass
(no cache, no batching), so 'noise alpha = 0' is bit-identical to 'normal'.
Variants per item: normal | disguise (N: fictional twin, R: symbol rewrite; C: none) | bang | wake | placebo | instr |
                   noise_<alpha> (residual of model.model.layers[20], every position incl. option tokens)
  modal run modal_dis.py --mode dev   --sets sets_dev.json --extra sets_smoke.json      (F2 on R-dev; alpha flip rates)
  modal run modal_dis.py --mode main  --sets sets_dis.json --alpha A --tag main          (smoke: --sets sets_smoke.json --tag smoke)
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-disguise", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
ALPHAS = [0.05, 0.1, 0.2, 0.4]
NOISE_LAYER = 20
R_TAIL = "\n\nReply with only the answer, nothing else."


def insert(it, line, prompt):
    if it["family"][0] == "R":
        return prompt.replace("Q:", line + "\nQ:", 1)
    return prompt.replace("Classify the TEST case below.", line + "\n\nClassify the TEST case below.", 1)


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=6 * 3600)
def run(mode: str, items: list, shakes: dict, alpha: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    D = model.config.hidden_size
    cur = {"alpha": 0.0, "seed": 0, "on": False}

    def hook(mod, inp, out):
        if not cur["on"]:
            return None
        h = out[0] if isinstance(out, tuple) else out
        T = h.shape[1]
        g = torch.Generator(device="cuda").manual_seed(cur["seed"])
        z = torch.randn((T, D), generator=g, device="cuda", dtype=torch.float32)
        hf = h.float()
        hn = hf + cur["alpha"] * hf.norm(dim=-1, keepdim=True) / D ** 0.5 * z.unsqueeze(0)
        hn = hn.to(h.dtype)
        return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn
    model.model.layers[NOISE_LAYER].register_forward_hook(hook)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    def render(it, prompt, cands):
        if it["family"][0] == "R":
            return chat(prompt + R_TAIL) + "The answer is", [" " + c for c in cands]
        return chat(prompt), list(cands)

    def score(prefix, opts, noise=None, seed=0):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; out = []
        cur.update(on=noise is not None, alpha=noise or 0.0, seed=seed)
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            out.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])   # per-token log-probs (sum / first / mean later)
        cur["on"] = False
        return out

    checks = dict(alpha0_max_diff=0.0, repeat_max_diff=0.0)
    alphas = ALPHAS if mode == "dev" else [alpha] if alpha else []
    for n, it in enumerate(items):
        res = {}
        pre, opts = render(it, it["prompt"], it["cands"])
        res["normal"] = score(pre, opts)
        seed = 12345 + int(it["uid"])
        if mode != "dev":
            if it.get("twin"):
                p2, o2 = render(it, it["twin"], it["twin_cands"]); res["disguise"] = score(p2, o2)
            elif it.get("disg") is not None:
                p2, o2 = render(it, it["disg"], it["cands"]); res["disguise"] = score(p2, o2)
            for k, line in shakes.items():
                p2, o2 = render(it, insert(it, line, it["prompt"]), it["cands"]); res[k] = score(p2, o2)
            extra = ALPHAS if it["family"][0] in "NR" else [alpha]
            alphas = sorted(set(extra) | {alpha})
        for a in alphas:
            res[f"noise_{a}"] = score(pre, opts, noise=a, seed=seed)
        if n < 10:
            z = score(pre, opts, noise=0.0, seed=seed)
            checks["alpha0_max_diff"] = max(checks["alpha0_max_diff"], max(abs(x - y) for u, v in zip(z, res["normal"]) for x, y in zip(u, v)))
            a = alphas[0] if alphas else 0.1
            r1, r2 = score(pre, opts, noise=a, seed=seed), score(pre, opts, noise=a, seed=seed)
            checks["repeat_max_diff"] = max(checks["repeat_max_diff"], max(abs(x - y) for u, v in zip(r1, r2) for x, y in zip(u, v)))
        meta = {k: v for k, v in it.items() if k not in ("prompt", "twin", "disg")}
        yield json.dumps(dict(meta, res=res))
        if (n + 1) % 100 == 0:
            print("dis", n + 1, "/", len(items), flush=True)
    yield json.dumps(dict(checks=checks))


@app.local_entrypoint()
def main(mode: str = "main", sets: str = "sets_smoke.json", extra: str = "", alpha: float = 0.0, tag: str = "smoke"):
    d = json.load(open(os.path.join(HERE, sets)))
    items = d["items"]
    if extra:
        items = [x for x in items] + [x for x in json.load(open(os.path.join(HERE, extra)))["items"] if x["family"].startswith("N")]
    for k, x in enumerate(items):
        x.setdefault("uid", 100000 + k)
    shakes = d.get("shakes") or json.load(open(os.path.join(HERE, "sets_smoke.json")))["shakes"]
    out = os.path.join(HERE, f"out_{mode}_{tag}.jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(mode, items, shakes, alpha):
            f.write(line + "\n"); f.flush()
    print("wrote", out)
