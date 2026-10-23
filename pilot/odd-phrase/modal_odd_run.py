"""Odd-phrase pilot, stages after text generation (see README.md).

  modal run modal_odd_run.py --stage swap        # splits.json -> swaps.json: one word swapped per subtle target
  python build_items.py                          # -> items.json
  modal run modal_odd_run.py --stage run --models A,B  [--dtype float32]   # -> results/<model>.json
  modal run modal_odd_run.py --stage judge       # results/*.json -> judged/<model>.json (Qwen2.5-7B as judge)

Conversation per item (greedy decoding):
  turn 1  user = the text only, no instruction          -> reply1 (did the model mention anything odd?)
  turn 2  user = ASK (find the strange phrases)         -> reply2 (which ones did it find?)
Implicit measure (separate pass, plain text): per-sentence mean token surprisal minus the same sentence's
surprisal in its own original paragraph (z, nats). Results return in gzip chunks under Modal's inline limit.
"""

import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("V2_GPU", "L4")
CHUNK = 1024 * 1024
JUDGE = "microsoft/Phi-3.5-mini-instruct"  # also writes the word swaps; not one of the tested models

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-odd-phrase-run", image=image)

ASK = ("Some phrases or words in this text may be strange: they do not belong to it. Find every such phrase or "
       "word. Quote each one exactly, one per line. If there are none, say NONE.")

SWAP_PROMPT = ("Rewrite the sentence below, replacing exactly ONE noun with a noun from the topic area '{far}' that "
               "makes no sense in this sentence. Keep every other word the same. Reply in this format and nothing "
               "else:\nWORD: <the new noun>\nSENTENCE: <the rewritten sentence>\n\nSentence: {s}")

JUDGE_PROMPT = ("A reader was given a text and replied to it. Did the reader's reply say that some part of the "
                "text was strange, out of place, unrelated, or did not belong? Answer YES or NO only.\n\n"
                "Reader's reply:\n{r}")


def _pack(obj):
    import gzip
    blob = gzip.compress(json.dumps(obj).encode())
    for i in range(0, len(blob), CHUNK):
        yield blob[i:i + CHUNK]


def _unpack(gen):
    import gzip
    return json.loads(gzip.decompress(b"".join(gen)))


def _load(name, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name).to(getattr(torch, dtype)).to("cuda").eval()
    return tok, model


def _chat_batch(tok, model, convs, max_new):
    """Greedy replies for a list of conversations (batched, left padding)."""
    import torch
    out = []
    for i in range(0, len(convs), 16):
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs[i:i + 16]]
        b = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**b, max_new_tokens=max_new, do_sample=False)
        out += [tok.decode(r[b["input_ids"].shape[1]:], skip_special_tokens=True).strip() for r in g]
    return out


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def swap_remote(targets: list):
    tok, model = _load(JUDGE, "bfloat16")
    convs = [[{"role": "user", "content": SWAP_PROMPT.format(far=t["far"], s=t["s"])}] for t in targets]
    yield from _pack(_chat_batch(tok, model, convs, 120))


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3 * 3600)
def run_remote(model_name: str, items: list, native_texts: dict, dtype: str = "float32"):
    import torch
    t0 = time.time()
    tok, model = _load(model_name, dtype)

    def sent_surprisal(sents):
        text = " ".join(sents)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0].float(), -1)
        nll = (-logp[:-1].gather(1, ids[0, 1:, None])[:, 0]).cpu().tolist()
        offs = enc["offset_mapping"][0].tolist()
        starts, pos = [], 0
        for s in sents:
            starts.append(pos)
            pos += len(s) + 1
        sums, cnts = [0.0] * len(sents), [0] * len(sents)
        for t in range(1, len(offs)):
            a, b = offs[t]
            j = max(k for k in range(len(sents)) if starts[k] <= max(a, (a + b) // 2))
            sums[j] += nll[t - 1]
            cnts[j] += 1
        return [s / c if c else None for s, c in zip(sums, cnts)]

    native = {tid: sent_surprisal(s) for tid, s in native_texts.items()}
    conv1 = [[{"role": "user", "content": " ".join(it["sentences"])}] for it in items]
    reply1 = _chat_batch(tok, model, conv1, 300)
    print(f"{model_name}: turn 1 done ({time.time() - t0:.0f} s)", flush=True)
    conv2 = [c + [{"role": "assistant", "content": r}, {"role": "user", "content": ASK}] for c, r in zip(conv1, reply1)]
    reply2 = _chat_batch(tok, model, conv2, 300)
    print(f"{model_name}: turn 2 done ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for it, r1, r2 in zip(items, reply1, reply2):
        surp = sent_surprisal(it["sentences"])
        z = [None if s is None or native[src[0]][src[1]] is None else s - native[src[0]][src[1]]
             for s, src in zip(surp, it["source"])]
        rows.append({"id": it["id"], "reply1": r1, "reply2": r2, "surp": surp, "z": z})
    cache.commit()
    yield from _pack({"model": model_name, "dtype": dtype, "rows": rows, "seconds": time.time() - t0})


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def judge_remote(replies: list):
    import torch
    tok, model = _load(JUDGE, "bfloat16")
    yes = tok.encode("YES", add_special_tokens=False)[0]
    no = tok.encode("NO", add_special_tokens=False)[0]
    out = []
    for i in range(0, len(replies), 16):
        prompts = [tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(r=r)}], tokenize=False,
                                           add_generation_prompt=True) for r in replies[i:i + 16]]
        b = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**b).logits[:, -1].float()
        out += (lg[:, yes] - lg[:, no]).cpu().tolist()  # > 0 = YES
    yield from _pack(out)


@app.local_entrypoint()
def main(stage: str = "run", models: str = "Qwen/Qwen2.5-1.5B-Instruct", dtype: str = "float32"):
    def rd(name):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            return json.load(f)

    def wr(name, obj):
        os.makedirs(os.path.dirname(os.path.join(HERE, name)), exist_ok=True)
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    if stage == "swap":
        targets = rd("splits.json")["swap_targets"]
        wr("swaps.json", _unpack(swap_remote.remote_gen(targets)))
        print(f"swaps for {len(targets)} sentences")
    elif stage == "run":
        data = rd("items.json")
        from concurrent.futures import ThreadPoolExecutor

        def one(m):
            res = _unpack(run_remote.remote_gen(m, data["items"], data["native_texts"], dtype))
            wr(f"results/{m.split('/')[-1]}.json", res)
            print(f"{m}: {len(res['rows'])} rows, {res['seconds'] / 60:.1f} min")

        names = [m.strip() for m in models.split(",") if m.strip()]
        with ThreadPoolExecutor(len(names)) as pool:
            list(pool.map(one, names))
    elif stage == "judge":
        for fn in sorted(os.listdir(os.path.join(HERE, "results"))):
            res = rd(f"results/{fn}")
            scores = _unpack(judge_remote.remote_gen([r["reply1"] for r in res["rows"]]))
            wr(f"judged/{fn}", {"model": res["model"], "judge": JUDGE, "reply1_yes_minus_no": scores})
            print(f"judged {fn}")
