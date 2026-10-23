"""Think-aloud test (PLAN.md v2 + v2.1) on Qwen2.5-14B-Instruct, bf16, greedy.
1) THINK and THINKI reasoning: batched greedy generation (left padding, batches sorted by length), max 512 new tokens.
2) Forced choice for every arm: each option its own full forward pass (no cache); per-token log-probs saved.
  modal run modal_ta.py --sets sets_smoke.json --tag smoke     (also: batched vs single greedy check on 20 prompts)
  modal run modal_ta.py --sets sets_ta.json --tag main
Hard cap: function timeout 3 h (H100 ~ $4/h -> <= ~$12)."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-thinkaloud", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
MAXNEW, BS = 512, 32


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(items: list, check_batch: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    def generate(prompts):
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                new = g[j, enc.input_ids.shape[1]:].tolist()
                trunc = not any(t in eos for t in new)
                out[i] = (tok.decode(new, skip_special_tokens=True), trunc)
            print("gen", min(b + BS, len(order)), "/", len(order), flush=True)
        return out

    def score(prefix, opts):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; res = []
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            res.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])
        return res

    def direct(it, text):
        if it["family"] == "N":
            return score(chat(text), list(it["cands"]))
        return score(chat(text + R_TAIL) + "The answer is", [" " + c for c in it["cands"]])

    def after(it, prompt, g):
        base = chat(prompt)
        pre = base + g[:g.rindex("Answer:")] + "Answer:" if "Answer:" in g else base + g.rstrip() + "\nAnswer:"
        return score(pre, [" " + c for c in it["cands"]])

    checks = {}
    if check_batch:
        ps = [chat(it["think"]) for it in items[:20]]
        bat = generate(ps)
        single = []
        for p in ps:
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            single.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        checks["batch_vs_single_identical"] = sum(a[0] == b for a, b in zip(bat, single))
        checks["batch_vs_single_same_answer_line"] = sum(a[0].strip().split("\n")[-1] == b.strip().split("\n")[-1] for a, b in zip(bat, single))
        checks["n_checked"] = len(ps)
    gt = generate([chat(it["think"]) for it in items])
    gi = generate([chat(it["thinki"]) for it in items])
    for n, it in enumerate(items):
        res = {"direct": direct(it, it["direct"]), "instr": direct(it, it["instr"])}
        if it.get("disg") is not None:
            res["disg"] = res["direct"] if it["disg"] == it["direct"] else direct(it, it["disg"])
        if it.get("truth"):
            res["truth"] = score(chat(it["truth"]) + "The answer is", [" yes", " no"])
        res["think"] = after(it, it["think"], gt[n][0]); res["thinki"] = after(it, it["thinki"], gi[n][0])
        meta = {k: v for k, v in it.items() if k not in ("direct", "instr", "think", "thinki", "disg", "truth")}
        yield json.dumps(dict(meta, res=res, gen_think=gt[n][0], trunc_think=gt[n][1], gen_thinki=gi[n][0], trunc_thinki=gi[n][1]))
        if (n + 1) % 100 == 0:
            print("score", n + 1, "/", len(items), flush=True)
    yield json.dumps(dict(checks=checks))


@app.local_entrypoint()
def main(sets: str = "sets_smoke.json", tag: str = "smoke"):
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    out = os.path.join(HERE, f"out_{tag}.jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(items, tag == "smoke"):
            f.write(line + "\n"); f.flush()
    print("wrote", out)
