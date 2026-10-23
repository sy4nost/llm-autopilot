"""Conflict lamp (Zhao et al. 2024 style linear probe) + gated 'use the prompt' brake on Qwen2.5-14B-Instruct.
1. Own facts (facts.json): hidden state at the last prompt token, every layer. Probe conflict vs consistent, trained on facts
   with index%3 != 0, chosen layer = best held-out accuracy on index%3 == 0.
2. Brake direction at that layer = mean(state | 'Use only the passage') - mean(state | 'Ignore the passage, use your own knowledge')
   on the training conflict facts. Brake = add alpha * direction at that layer from the last prompt token onward.
3. Score every item normal and with the brake at each alpha; lamp = probe on the normal state. Gated = brake only when lamp on.
   Items: own facts, inverse-scaling redefine-math / quote-repetition / hindsight-neglect-10shot (200 each), MMLU 0-shot (200).
  modal run modal_conflict.py   -> conflict_<model>.jsonl (one row per item, plus a 'meta' row)
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-conflict14", image=image)
ALPHAS = [1.0, 2.0, 4.0]
USE = "Use only the passage to answer, even if it disagrees with what you know.\n\n"
IGN = "Ignore the passage and answer from your own knowledge.\n\n"


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(model_name: str, facts: list, n_pub: int):
    import random, torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    NL = model.config.num_hidden_layers
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    st = {"vec": None, "L": None, "t0": 0}

    def hook(mod, args, out):
        if st["vec"] is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        start = 0 if h.shape[1] == 1 else st["t0"]
        h[:, start:] += st["vec"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def states(text):  # last-token state after each layer (index 1..NL)
        ids = tok(chat(text), add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        return torch.stack([h[0, -1].float() for h in hs[1:]])  # [NL, d]

    def score(text, cands, vec=None):
        pid = tok(chat(text), add_special_tokens=False)["input_ids"]
        st["t0"], st["vec"] = len(pid) - 1, vec
        sc = []
        for c in cands:
            ids_c = tok(c.strip(), add_special_tokens=False)["input_ids"]
            ids = torch.tensor([pid + ids_c]).cuda()
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=len(ids_c) + 1).logits[0, :-1].float()
            sc.append(round(torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c).cuda().unsqueeze(-1)).sum().item(), 3))
        st["vec"] = None
        return sc
    # ---- 1. probe on own facts
    idx = {f["id"] for f in facts}
    test_ids = {i for k, i in enumerate(sorted(idx)) if k % 3 == 0}
    S = torch.stack([states(f["prompt"]) for f in facts])  # [N, NL, d]
    y = torch.tensor([1.0 if f["version"] == "conflict" else 0.0 for f in facts]).cuda()
    tr = torch.tensor([f["id"] not in test_ids for f in facts]).cuda()
    print("states", S.shape, flush=True)

    def fit(X, Y, steps=500):
        mu, sd = X.mean(0), X.std(0) + 1e-4
        Z = (X - mu) / sd
        w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, Y) + 1e-2 * (w ** 2).mean()
            loss.backward()
            opt.step()
        return lambda Xn: torch.sigmoid(((Xn - mu) / sd) @ w.detach() + b.detach())
    acc = []
    for L in range(NL):
        p = fit(S[tr, L], y[tr])
        acc.append(((p(S[~tr, L]) > 0.5).float() == y[~tr]).float().mean().item())
    Lb = max(range(8, NL), key=lambda L: acc[L])
    probe = fit(S[tr, Lb], y[tr])
    print("probe acc by layer", [round(a, 2) for a in acc], "best", Lb, flush=True)
    # ---- 2. brake direction from training conflict facts
    cf = [f for f in facts if f["version"] == "conflict" and f["id"] not in test_ids]
    du = torch.stack([states(USE + f["prompt"])[Lb] for f in cf]).mean(0)
    di = torch.stack([states(IGN + f["prompt"])[Lb] for f in cf]).mean(0)
    direction = (du - di).to("cuda")
    model.model.layers[Lb].register_forward_hook(hook)
    yield json.dumps(dict(meta=True, layer=Lb + 1, probe_acc=[round(a, 3) for a in acc], dir_norm=direction.norm().item()))
    # ---- 3. evaluate
    items = [dict(task="own-" + f["version"], i=f["id"], test=f["id"] in test_ids, prompt=f["prompt"], cands=f["cands"], gold=0) for f in facts]
    for name in ("redefine-math", "quote-repetition", "hindsight-neglect-10shot"):
        ds = load_dataset("inverse-scaling/" + name, split="train")
        ix = list(range(len(ds)))
        random.Random(0).shuffle(ix)
        for n, i in enumerate(sorted(ix[:n_pub])):
            r = ds[i]
            items.append(dict(task=name, i=n, test=True, prompt=r["prompt"].replace("\r\n", "\n"), cands=[c.strip() for c in r["classes"]], gold=r["answer_index"]))
    mm = load_dataset("cais/mmlu", "all", split="test")
    ix = list(range(len(mm)))
    random.Random(0).shuffle(ix)
    for n, i in enumerate(sorted(ix[:n_pub])):
        r = mm[i]
        q = "Answer with the letter only.\n\n" + r["question"].strip() + "\n" + "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(r["choices"])) + "\nAnswer:"
        items.append(dict(task="mmlu", i=n, test=True, prompt=q, cands=list("ABCD"), gold=r["answer"]))
    for k, it in enumerate(items):
        lamp = probe(states(it["prompt"])[Lb].unsqueeze(0)).item()
        row = dict(task=it["task"], i=it["i"], test=it["test"], gold=it["gold"], lamp=round(lamp, 4), normal=score(it["prompt"], it["cands"]))
        for a in ALPHAS:
            row[f"brake{a:g}"] = score(it["prompt"], it["cands"], a * direction)
        yield json.dumps(row)
        if (k + 1) % 50 == 0:
            print("eval", k + 1, "/", len(items), flush=True)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n_pub: int = 200, tag: str = ""):
    facts = json.load(open(os.path.join(HERE, "facts.json")))
    out = os.path.join(HERE, "conflict_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, facts, n_pub):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
