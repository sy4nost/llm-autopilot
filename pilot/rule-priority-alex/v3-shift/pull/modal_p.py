"""pull scoring: for each prompt (chat template + generation prompt) and each of its 3 candidate numbers, log P(candidate
tokens) + log P(next token does not start with a digit) -> probability that the whole answer starts with that number.
  modal run modal_p.py --tag smoke --n 60 ;  modal run modal_p.py --tag smoke --get"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-pull", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
BS = 24


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str, tag: str):
    import torch, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    tok = AutoTokenizer.from_pretrained(MODELS[mk])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    V = model.get_output_embeddings().weight.shape[0]
    digit = torch.zeros(V, dtype=torch.bool)
    for t in range(min(V, len(tok))):
        s = tok.decode([t]).lstrip(" ")
        digit[t] = bool(s) and s[0].isdigit()
    digit = digit.to("cuda")
    rows = []
    for it in items:
        p = tok(tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        for src, v in it["cand"].items():
            rows.append((it["id"], src, p, tok(str(v), add_special_tokens=False)["input_ids"], "bare"))
            rows.append((it["id"], src, p, tok(it["sent"][src], add_special_tokens=False)["input_ids"], "sent"))
    print(mk, "rows", len(rows), "example cand tokens", [tok.decode([t]) for t in rows[0][3]], flush=True)
    rows.sort(key=lambda r: len(r[2]) + len(r[3])); out = {}
    d = "/cache/pull"; os.makedirs(d, exist_ok=True)
    for b in range(0, len(rows), BS):
        R = rows[b:b + BS]; ml = max(len(p) + len(c) for _, _, p, c, _ in R)
        ids = torch.full((len(R), ml), pad); att = torch.zeros((len(R), ml), dtype=torch.long)
        for j, (_, _, p, c, _) in enumerate(R):
            ids[j, :len(p) + len(c)] = torch.tensor(p + c); att[j, :len(p) + len(c)] = 1
        with torch.no_grad():
            lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        for j, (iid, src, p, c, form) in enumerate(R):
            lp = torch.log_softmax(lg[j, len(p) - 1:len(p) + len(c)].float(), -1)   # only the needed positions
            s = sum(lp[k, c[k]].item() for k in range(len(c)))
            pd = lp[len(c)].exp()[digit].sum().clamp(max=1 - 1e-9).item()
            out.setdefault(iid, {})[f"{src}|{form}"] = s + float(torch.log1p(torch.tensor(-pd))) if form == "bare" else s
        print(mk, min(b + BS, len(rows)), "/", len(rows), flush=True)
    open(f"{d}/{tag}_{mk}.json", "w").write(json.dumps(out)); open(f"{d}/{tag}_{mk}.DONE", "w").write("1"); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, mk: str):
    cache.reload(); d = "/cache/pull"
    return json.dumps(dict(out=open(f"{d}/{tag}_{mk}.json").read(), done=os.path.exists(f"{d}/{tag}_{mk}.DONE")))


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False, only: str = ""):
    import gzip, base64
    mks = only.split(",") if only else list(MODELS)
    if get:
        for mk in mks:
            try:
                r = json.loads(fetch.remote(tag, mk)); open(os.path.join(HERE, f"out_{tag}_{mk}.json"), "w").write(r["out"])
                print(mk, "ok", "done" if r["done"] else "PARTIAL", len(json.loads(r["out"])))
            except Exception as e:
                print(mk, "not fetched", repr(e)[:150])
        return
    items = json.load(open(os.path.join(HERE, "sets_p.json")))["items"]
    if n:
        import random
        keep = set(random.Random(2).sample(sorted({it["src_id"] for it in items}), n // 3))
        items = [it for it in items if it["src_id"] in keep]
    blob = base64.b64encode(gzip.compress(json.dumps([dict(id=it["id"], cand=it["cand"], sent=it["sent"], messages=it["messages"]) for it in items]).encode())).decode()
    calls = {mk: run.spawn(blob, mk, tag) for mk in mks}
    for mk, c in calls.items():
        try:
            print(mk, c.get())
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
