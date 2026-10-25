"""Surprise of every task word (surprise-lamp/PLAN.md). One forward pass per item per model, no generation.
Text = the user turn exactly as DIRECT saw it in conftrap-lamp (mode q: prompt + R_TAIL; label: prompt), inside the chat template.
Only tokens whose characters lie inside the user content are kept: [char start, char end, surprisal, model's top-1 guess].
  modal run --detach modal_sp.py --sets ../conftrap-lamp/sets_cl.json --tag main --models qwen14   (one launch per model)
  modal run modal_sp.py --tag main --get        (writes sp_<tag>_<model>.jsonl)"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-surprise", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
R_TAIL = "\n\nReply with only the answer, nothing else."   # as conftrap-lamp/modal_cl.py


def content(it):
    return it["prompt"] if it["mode"] == "label" else it["prompt"] + R_TAIL


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=2 * 3600)
def run(items: list, tag: str, mk: str):
    import torch, time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[mk]; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    # surprise "given that the text goes on": special tokens (end of turn etc.) get no probability; smoke showed chat models
    # expect the turn to end after every complete question, which swamped the word-level surprise
    spec = sorted({i for i in tok.all_special_ids} | {i for i, t in tok.added_tokens_decoder.items() if t.special})
    d = f"/cache/sp_runs/{tag}/{mk}"; os.makedirs(d, exist_ok=True); ofn = f"{d}/out.jsonl"
    done = set()
    if os.path.exists(ofn):
        good = []
        for l in open(ofn):
            try:
                x = json.loads(l)
                if "tok" in x:
                    good.append(l if l.endswith("\n") else l + "\n"); done.add(x["id"])
            except ValueError:
                pass
        open(ofn, "w").writelines(good)
    of = open(ofn, "a"); bad = 0
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        c = content(it); s = chat(c); a = s.find(c)
        if a < 0:   # template changed the text: record and skip (counted in checks)
            bad += 1; of.write(json.dumps(dict(id=it["id"], model=mk, tok=None, err="content not found")) + "\n"); continue
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids]).cuda()).logits[0].float()
        lg[:, spec] = -float("inf"); lp = torch.log_softmax(lg[:-1], -1)
        sur = (-lp.gather(1, torch.tensor(ids[1:]).cuda()[:, None])[:, 0]).tolist()
        top = lp.argmax(-1).tolist()
        rows = []
        for i in range(1, len(ids)):
            st, en = offs[i]
            if st >= a and en <= a + len(c) and en > st:
                rows.append([st - a, en - a, round(sur[i - 1], 4), tok.decode([top[i - 1]])])
        of.write(json.dumps(dict(id=it["id"], model=mk, tok=rows)) + "\n")
        if (n + 1) % 50 == 0:
            of.flush(); cache.commit(); print(mk, n + 1, "/", len(items), flush=True)
    of.write(json.dumps(dict(checks=dict(model=name, not_found=bad, seconds=round(time.time() - t0),
                                         template=chat("Hi"), n_special=len(spec)))) + "\n")
    of.close(); cache.commit(); print(mk, "DONE", flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str, mk: str):
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/sp_runs/{tag}/{mk}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(sets: str = "../conftrap-lamp/sets_cl.json", tag: str = "smoke", get: bool = False, models: str = "qwen14,phi4,olmo13", n: int = 0):
    mks = models.split(",")
    if get:
        import gzip, base64
        for mk in mks:
            try:
                data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag, mk))))
            except Exception as e:
                print(mk, "not fetched:", repr(e)[:200]); continue
            out = os.path.join(HERE, f"sp_{tag}_{mk}.jsonl"); open(out, "wb").write(data); print("wrote", out, data.count(b"\n"))
        return
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    if n:   # smoke: n items spread over families
        items = items[::max(1, len(items) // n)][:n]
    calls = [run.spawn(items, tag, mk) for mk in mks]
    for mk, c in zip(mks, calls):
        try:
            c.get(); print(mk, "finished")
        except Exception as e:
            print(mk, "FAILED:", repr(e)[:300])
