"""Fast (DIRECT) forced-choice answers only, 3 models, bf16, L40S. No thinking, no lamp.
  modal run --detach modal_ct.py --tag pilot --models qwen14     (one launch per model)
  modal run modal_ct.py --tag pilot --get                         (writes out_<tag>_<model>.jsonl)
State on the volume /cache/wp_runs/<tag>/<model>/out.jsonl, committed every 50 items; a rerun skips saved items."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-wpub", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
R_TAIL = "\n\nReply with only the answer, nothing else."


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=2 * 3600)
def run(items: list, tag: str, mk: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[mk]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    def score(prefix, opts):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; res = []
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            res.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])
        return res

    def direct(it):
        if it["mode"] == "label":   # prompts that already end with "Return only the label/answer."
            return score(chat(it["prompt"]), list(it["cands"]))
        return score(chat(it["prompt"] + R_TAIL) + (it["stem"].strip() if it["stem"] else "The answer is"), [" " + c for c in it["cands"]])

    d = f"/cache/wp_runs/{tag}/{mk}"; os.makedirs(d, exist_ok=True); ofn = f"{d}/out.jsonl"
    done, good = set(), []
    if os.path.exists(ofn):
        for l in open(ofn):
            try:
                x = json.loads(l)
                if "res" in x:
                    done.add(x["id"]); good.append(json.dumps(x) + "\n")
            except ValueError:
                pass
        open(ofn, "w").writelines(good)
    of = open(ofn, "a")
    print(mk, "already", len(done), "of", len(items), flush=True)
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        meta = {k: v for k, v in it.items() if k != "prompt"}
        of.write(json.dumps(dict(meta, model=mk, res=direct(it))) + "\n")
        if (n + 1) % 50 == 0:
            of.flush(); cache.commit(); print(mk, "score", n + 1, "/", len(items), flush=True)
    of.write(json.dumps(dict(checks=dict(model=name, template_tail=chat("Hi")[-120:]))) + "\n"); of.close(); cache.commit()
    print(mk, "DONE", flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str, mk: str):
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/wp_runs/{tag}/{mk}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(sets: str = "sets_wp_pilot.json", tag: str = "pilot", get: bool = False, models: str = "qwen14,phi4,olmo13"):
    mks = models.split(",")
    if get:
        import gzip, base64
        for mk in mks:
            out = os.path.join(HERE, f"out_{tag}_{mk}.jsonl")
            try:
                data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag, mk))))
            except Exception as e:
                print(mk, "not fetched:", repr(e)[:200]); continue
            open(out, "wb").write(data); print("wrote", out, data.count(b"\n"), "lines")
        return
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    for mk in mks:
        run.remote(items, tag, mk); print(mk, "finished")
