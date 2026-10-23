"""Gurdjieff sensor (PLAN.md v2 + v2.1): read Qwen2.5-14B's residual stream at the DIRECT decision position, then train/test
sensors on Modal CPU. State lives on the volume (/cache/gurdjieff/<tag>/), so a lost client loses nothing.
  modal run --detach modal_g.py --tag smoke --n 40        extraction on the first 40 items of each pool (+ sanity per family)
  modal run --detach modal_g.py --tag main                extraction on all items
  modal run modal_g.py --tag main --analyze               sensors + verdict on CPU; writes report_<tag>.txt, scores_<tag>.jsonl
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "numpy")
             .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2")
             .add_local_file(os.path.join(HERE, "analyze_g.py"), "/root/analyze_g.py"))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-gurdjieff")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]


@app.function(image=gpu_image, gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def extract(items: list, tag: str):
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    d = f"/cache/gurdjieff/{tag}"; os.makedirs(d, exist_ok=True)
    ffn = f"{d}/feats.jsonl"
    done, good = set(), []
    if os.path.exists(ffn):  # keep only whole lines, so a cut last line cannot merge with the next write
        for l in open(ffn):
            try:
                x = json.loads(l); done.add(x["id"]); good.append(json.dumps(x) + "\n")
            except ValueError:
                pass
        open(ffn, "w").writelines(good)
    ff = open(ffn, "a")
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        # verbatim copies of each run's direct(): own N = chat(text) + bare options; else chat(text + tail) + cue + " "+option
        prefix = chat(it["user"]) if it["mode"] == "raw" else chat(it["user"] + R_TAIL) + it["cue"]
        pid = tok(prefix, add_special_tokens=False)["input_ids"]
        oids = [tok(o, add_special_tokens=False)["input_ids"] for o in it["opts"]]
        k = 0  # shared leading option tokens belong to the prefix; decision = first token where options differ
        while all(len(o) > k for o in oids) and len({o[k] for o in oids}) == 1:
            k += 1
        sums, hid, dec = [], None, None
        for j, o in enumerate(oids):
            ids = torch.tensor([pid + o]).cuda()
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=(j == 0))
            if j == 0:
                assert len(out.hidden_states) == model.config.num_hidden_layers + 1
            lp = torch.log_softmax(out.logits[0, len(pid) - 1:-1].float(), -1)
            sums.append(round(sum(lp[i, t].item() for i, t in enumerate(o)), 4))
            if j == 0:
                pos = len(pid) + k - 1  # this position's logits predict the first differing option token
                hid = np.stack([out.hidden_states[L][0, pos].float().cpu().numpy() for L in LAYERS]).astype(np.float32)
                assert np.isfinite(hid).all()
                dlp = torch.log_softmax(out.logits[0, pos].float(), -1)
                dec = [round(dlp[oo[k]].item(), 4) if len(oo) > k else 0.0 for oo in oids]
        np.save(f"{d}/h_{it['id']}.npy", hid)
        ff.write(json.dumps(dict(id=it["id"], sums=sums, dec=dec, pick=int(np.argmax(sums)), k=k)) + "\n")
        if (n + 1) % 50 == 0:
            ff.flush(); cache.commit(); print("extract", n + 1, "/", len(items), flush=True)
    ff.close(); cache.commit(); print("DONE extract", tag, flush=True)


@app.function(image=cpu_image, volumes={"/cache": cache}, timeout=2 * 3600, cpu=8, memory=32768)
def analyze_remote(meta: list, tag: str):
    import sys
    sys.path.insert(0, "/root")
    import numpy as np, analyze_g
    cache.reload()
    d = f"/cache/gurdjieff/{tag}"
    feats = {}
    for l in open(f"{d}/feats.jsonl"):
        try:
            x = json.loads(l); feats[x["id"]] = x
        except ValueError:
            pass
    missing = [m["id"] for m in meta if m["id"] not in feats]
    print("missing items", len(missing), flush=True)
    meta = [m for m in meta if m["id"] in feats]
    H = np.stack([np.load(f"{d}/h_{m['id']}.npy") for m in meta]).astype(np.float32)
    report, scores = analyze_g.run(meta, feats, H, LAYERS, n_missing=len(missing))
    open(f"{d}/report.txt", "w").write(report)
    with open(f"{d}/scores.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")
    cache.commit()
    return dict(report=report, scores=scores)


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, analyze: bool = False):
    items = json.load(open(os.path.join(HERE, "items_g.json")))["items"]
    if n:
        items = [x for p in ("own", "pub") for x in [y for y in items if y["pool"] == p][::max(1, 2060 // n)][:n]]
    if analyze:
        meta = [{k: v for k, v in x.items() if k not in ("user", "opts")} for x in items]
        out = analyze_remote.remote(meta, tag)
        open(os.path.join(HERE, f"report_{tag}.txt"), "w").write(out["report"])
        with open(os.path.join(HERE, f"scores_{tag}.jsonl"), "w") as f:
            for s in out["scores"]:
                f.write(json.dumps(s) + "\n")
        print(out["report"]); return
    extract.remote(items, tag)

