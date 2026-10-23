"""Mirror test (PLAN.md). State on the volume /cache/gurdjieff_mirror/<tag>/, resumable.
  modal run mirror_g.py --lamp-only                         CPU: out-of-fold lamp scores -> lamp_scores.json (freeze before GPU)
  modal run --detach mirror_g.py --tag smoke --n 20    GPU forced choice, first n items per family x all lines
  modal run --detach mirror_g.py --tag main            all items
  modal run mirror_g.py --tag main --fetch               fetch scores -> out_<tag>.jsonl
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "numpy")
             .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2")
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-gurdjieff-mirror")
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]
FAMS = ["R", "D"]
def ckey(x):  # R: arithmetic expression (twins together); D: concept cluster
    import re
    if x["fam"] == "R":
        m = re.search(r"\d+\s*(?:[-+*x×/]|mod)\s*\d+", x["user"])
        return "R:" + (re.sub(r"\s+", "", m.group(0)) if m else x["id"])
    return x["cluster"]




@app.function(image=cpu_image, volumes={"/cache": cache}, cpu=8, memory=16384, timeout=3600)
def lamp(meta: list):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    for f in FAMS:
        ms = [m for m in meta if m["fam"] == f]
        X = np.stack([np.load(f"/cache/gurdjieff/main/h_{m['id']}.npy")[LAYERS.index(32)] for m in ms]).astype(np.float32)
        y = np.array([m["y"] for m in ms]); g = np.array([m["cluster"] for m in ms]); s = np.zeros(len(ms))
        for tr, te in GroupKFold(5).split(X, y, g):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
            s[te] = mdl.decision_function(X[te])
        t = float(np.quantile(s, 0.70))
        for m, v in zip(ms, s):
            out[m["id"]] = dict(score=round(float(v), 4), fires=bool(v > t))
    return out


@app.function(image=gpu_image, gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def score(jobs: list, tag: str, lines: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    d = f"/cache/gurdjieff_mirror/{tag}"; os.makedirs(d, exist_ok=True); fn = f"{d}/scores.jsonl"
    done, good = set(), []
    if os.path.exists(fn):
        for l in open(fn):
            try:
                x = json.loads(l); done.add((x["id"], x["line"])); good.append(json.dumps(x) + "\n")
            except ValueError:
                pass
        open(fn, "w").writelines(good)
    f = open(fn, "a")
    for n, j in enumerate(jobs):
        if (j["id"], j["line"]) in done:
            continue
        line = lines[j["line"]]
        user = j["user"] + ("\n\n" + line if line else "")
        pid = tok(chat(user + R_TAIL) + j["cue"], add_special_tokens=False)["input_ids"]
        sums = []
        for o in j["opts"]:
            oid = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + oid]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            sums.append(round(sum(lp[i, t].item() for i, t in enumerate(oid)), 4))
        f.write(json.dumps(dict(id=j["id"], line=j["line"], sums=sums)) + "\n")
        if (n + 1) % 100 == 0:
            f.flush(); cache.commit(); print("scored", n + 1, "/", len(jobs), flush=True)
    f.close(); cache.commit(); print("DONE", tag, flush=True)


@app.function(image=cpu_image, volumes={"/cache": cache}, timeout=600)
def get(tag: str):
    cache.reload()
    yield open(f"/cache/gurdjieff_mirror/{tag}/scores.jsonl").read()


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, lamp_only: bool = False, fetch: bool = False):
    items = [x for x in json.load(open(os.path.join(HERE, "..", "items_g.json")))["items"] if x["pool"] == "own" and x["fam"] in FAMS]
    if lamp_only:
        meta = [dict(id=x["id"], fam=x["fam"], y=x["y"], cluster=ckey(x)) for x in items]
        json.dump(lamp.remote(meta), open(os.path.join(HERE, "lamp_scores.json"), "w"), indent=0); print("lamp saved"); return
    if fetch:
        with open(os.path.join(HERE, f"out_{tag}.jsonl"), "w") as f:
            for chunk in get.remote_gen(tag):
                f.write(chunk)
        print("fetched"); return
    if n:
        items = [x for fm in FAMS for x in [y for y in items if y["fam"] == fm][::max(1, 1026 // n)][:n]]
    lines = json.load(open(os.path.join(HERE, "lines.json")))  # name -> text ("" = no line); read locally only
    jobs = [dict(id=x["id"], user=x["user"], cue=x["cue"], opts=x["opts"], line=k) for x in items for k in lines]
    print(len(jobs), "jobs"); score.remote(jobs, tag, lines)
