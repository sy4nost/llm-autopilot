"""Odd-phrase pilot on Modal: (1) generate host texts with a non-Qwen author model (Phi-3.5-mini) so no tested model wrote them, (2) score items with several models.

  modal run modal_odd.py --stage gen                  # -> texts.json (60 topics x 2 paragraphs)
  python build_items.py                               # texts.json -> items.json (local, no GPU)
  modal run modal_odd.py --stage run --models Qwen/Qwen2.5-1.5B-Instruct,...   # -> results/<model>.json

Per item the model sees the text twice, in separate passes:
  explicit: numbered sentences + "which sentence does not belong? number or 0"; readout = probabilities of the
            digit tokens 0..n at the first answer position (no sampling).
  implicit: the plain text with no instruction; per-sentence mean token surprisal (nats). Each sentence's
            surprisal is compared with the same sentence in its own original paragraph (z = in doc - native).
Env: V2_GPU (default L4). Results return in gzip chunks under Modal's inline limit (blob host is blocked here).
"""

import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("V2_GPU", "L4")
CHUNK = 1024 * 1024

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-odd-phrase", image=image)


def _pack(obj):
    import gzip
    blob = gzip.compress(json.dumps(obj).encode())
    for i in range(0, len(blob), CHUNK):
        yield blob[i:i + CHUNK]


def _unpack(gen):
    import gzip
    return json.loads(gzip.decompress(b"".join(gen)))


GEN_PROMPT = ("Write one paragraph of exactly 8 sentences about {sub} (topic area: {dom}). Use a plain, factual, "
              "encyclopedic style. Every sentence must make sense on its own and must not start with a pronoun "
              "such as It, This, That, They, These, He or She. No lists, no headings, no title, only the paragraph.")


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def gen_remote(topics: list, per_topic: int = 2):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = os.environ.get("ODD_AUTHOR", "microsoft/Phi-3.5-mini-instruct")
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name).to(torch.bfloat16).to("cuda").eval()
    prompts = []
    for dom, sub in topics:
        msgs = [{"role": "user", "content": GEN_PROMPT.format(sub=sub, dom=dom)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    out = []
    for rep in range(per_topic):
        torch.manual_seed(100 + rep)
        for i in range(0, len(prompts), 20):
            batch = tok(prompts[i:i + 20], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = model.generate(**batch, max_new_tokens=400, do_sample=True, temperature=0.8, top_p=0.95)
            for k, row in enumerate(g):
                text = tok.decode(row[batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                dom, sub = topics[i + k]
                out.append({"domain": dom, "sub": sub, "rep": rep, "text": text})
        print(f"rep {rep}: {len(out)} texts", flush=True)
    cache.commit()
    yield from _pack(out)


EXPLICIT = ("Read the text below. Its sentences are numbered. One sentence may have been inserted from a "
            "different text. Which sentence does not belong? Reply with the number only, or 0 if every sentence "
            "belongs.\n\n{body}")


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3 * 3600)
def run_remote(model_name: str, items: list, texts: dict, dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(getattr(torch, dtype)).to("cuda").eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(x) == 1 for x in digit_ids), digit_ids
    digit_ids = [x[0] for x in digit_ids]

    def sent_surprisal(sents):
        text = " ".join(sents)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0].float(), -1)
        nll = -logp[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().tolist()  # nll of token t+1
        offs = enc["offset_mapping"][0].tolist()
        bounds, pos = [], 0
        for s in sents:
            bounds.append((pos, pos + len(s)))
            pos += len(s) + 1
        sums, cnts = [0.0] * len(sents), [0] * len(sents)
        for t in range(1, len(offs)):
            a, b = offs[t]
            mid = (a + b) / 2
            for j, (lo, hi) in enumerate(bounds):
                if lo <= mid < hi + 1:
                    sums[j] += nll[t - 1]
                    cnts[j] += 1
                    break
        return [s / c if c else float("nan") for s, c in zip(sums, cnts)]

    native = {tid: sent_surprisal(sents) for tid, sents in texts.items()}
    rows = []
    for k, it in enumerate(items):
        sents = it["sentences"]
        body = "\n".join(f"[{j + 1}] {s}" for j, s in enumerate(sents))
        msgs = [{"role": "user", "content": EXPLICIT.format(body=body)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = ids["input_ids"].to("cuda")
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        n = len(sents)
        dlp = [lp[digit_ids[d]].item() for d in range(min(n, 9) + 1)]
        surp = sent_surprisal(sents)
        z = []
        for j, src in enumerate(it["source"]):  # source = [text_id, index] of each sentence in its own paragraph
            z.append(surp[j] - native[src[0]][src[1]])
        rows.append({"id": it["id"], "dlp": dlp, "digit_mass": float(sum(torch.exp(torch.tensor(dlp)))),
                     "surp": surp, "z": z})
        if k % 50 == 0:
            print(f"{model_name}: item {k}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
    cache.commit()
    yield from _pack({"model": model_name, "dtype": dtype, "rows": rows, "seconds": time.time() - t0,
                      "native": native})


@app.local_entrypoint()
def main(stage: str = "run", models: str = "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct",
         dtype: str = "float32"):
    if stage == "gen":
        import sys
        sys.path.insert(0, HERE)
        from topics import TOPICS
        out = _unpack(gen_remote.remote_gen([list(t) for t in TOPICS]))
        with open(os.path.join(HERE, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"wrote {len(out)} texts")
        return
    with open(os.path.join(HERE, "items.json"), encoding="utf-8") as f:
        data = json.load(f)
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    names = [m.strip() for m in models.split(",") if m.strip()]

    def one(m):
        res = _unpack(run_remote.remote_gen(m, data["items"], data["texts"], dtype))
        path = os.path.join(HERE, "results", m.split("/")[-1] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        print(f"{m}: {len(res['rows'])} rows, {res['seconds'] / 60:.1f} min -> {path}")

    with ThreadPoolExecutor(len(names)) as pool:
        list(pool.map(one, names))
