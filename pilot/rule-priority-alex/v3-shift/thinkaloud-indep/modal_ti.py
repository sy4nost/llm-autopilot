"""Think-aloud on independent public tasks (thinkaloud-indep/PLAN.md); same code paths as thinkaloud/modal_ta.py. Qwen2.5-14B-Instruct, bf16, greedy.
1) THINK and THINKI reasoning: batched greedy generation (left padding, batches sorted by length), max 512 new tokens.
2) Forced choice for every arm: each option its own full forward pass (no cache); per-token log-probs saved.
  modal run modal_ti.py --sets sets_smoke.json --tag smoke     (also: batched vs single greedy check on 20 prompts)
  modal run modal_ti.py --sets sets_ti.json --tag main
Hard cap: function timeout 3 h (H100 ~ $4/h -> <= ~$12)."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-thinkaloud-indep", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
MAXNEW, BS = 512, 32


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(items: list, check_batch: bool, tag: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    def gen_iter(prompts):
        """yields (index, text, truncated) batch by batch, batches sorted by prompt length"""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                new = g[j, enc.input_ids.shape[1]:].tolist()
                yield i, tok.decode(new, skip_special_tokens=True), not any(t in eos for t in new)
            print("gen", min(b + BS, len(order)), "/", len(order), flush=True)

    def generate(prompts):
        out = [None] * len(prompts)
        for i, t, tr in gen_iter(prompts):
            out[i] = (t, tr)
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
        # completion-style tasks: the assistant turn is prefilled with the unfinished text, the option completes it
        return score(chat(text + R_TAIL) + (it["stem"].strip() if it["stem"] else "The answer is"), [" " + c for c in it["cands"]])

    def after(it, prompt, g):
        base = chat(prompt)
        pre = base + g[:g.rindex("Answer:")] + "Answer:" if "Answer:" in g else base + g.rstrip() + "\nAnswer:"
        return score(pre + (" " + it["stem"].strip() if it["stem"] else ""), [" " + c for c in it["cands"]])

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
    # resumable and client-independent: state lives on the volume (/cache/ti_runs/<tag>/), committed after every batch
    d = f"/cache/ti_runs/{tag}"; os.makedirs(d, exist_ok=True)
    gfn, ofn = f"{d}/gen.jsonl", f"{d}/out.jsonl"
    gens = {"think": {}, "thinki": {}}
    if os.path.exists(gfn):
        for l in open(gfn):
            try:
                g = json.loads(l)
            except ValueError:  # a line cut by a crash
                continue
            gens[g["gen"]][g["id"]] = (g["text"], g["trunc"])
    have_scored = [json.loads(l)["id"] for l in open(ofn) if l.endswith("}\n") and '"res"' in l] if os.path.exists(ofn) else []
    gf, of = open(gfn, "a"), open(ofn, "a")
    for a in ("think", "thinki"):
        todo = [it for it in items if it["id"] not in gens[a]]
        print(a, "already", len(gens[a]), "to generate", len(todo), flush=True)
        for k, (i, t, tr) in enumerate(gen_iter([chat(it[a]) for it in todo])):
            gens[a][todo[i]["id"]] = (t, tr)
            gf.write(json.dumps(dict(gen=a, id=todo[i]["id"], text=t, trunc=tr)) + "\n")
            if (k + 1) % BS == 0 or k + 1 == len(todo):
                gf.flush(); cache.commit()
    done = set(have_scored)
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        gt = {n: gens["think"][it["id"]]}; gi = {n: gens["thinki"][it["id"]]}
        res = {"direct": direct(it, it["direct"]), "instr": direct(it, it["instr"]),
               "rawc": score(it["raw"], list(it["raw_classes"]))}  # raw completion, no chat (sensitivity)
        res["think"] = after(it, it["think"], gt[n][0]); res["thinki"] = after(it, it["thinki"], gi[n][0])
        meta = {k: v for k, v in it.items() if k not in ("direct", "instr", "think", "thinki", "raw")}
        of.write(json.dumps(dict(meta, res=res, gen_think=gt[n][0], trunc_think=gt[n][1], gen_thinki=gi[n][0], trunc_thinki=gi[n][1])) + "\n")
        if (n + 1) % 100 == 0:
            of.flush(); cache.commit(); print("score", n + 1, "/", len(items), flush=True)
    of.write(json.dumps(dict(checks=checks)) + "\n"); of.close(); gf.close(); cache.commit()
    print("DONE", ofn, flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str):
    """returns out.jsonl from the volume in <1 MiB chunks (the proxy blocks Modal's large-blob host)"""
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/ti_runs/{tag}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(sets: str = "sets_smoke.json", tag: str = "smoke", get: bool = False):
    """launch (use `modal run --detach`, results stay on the volume) or, with --get, download out_{tag}.jsonl"""
    out = os.path.join(HERE, f"out_{tag}.jsonl")
    if get:
        import gzip, base64
        data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag))))
        open(out, "wb").write(data); print("wrote", out, data.count(b"\n"), "lines"); return
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    run.remote(items, tag == "smoke", tag)
