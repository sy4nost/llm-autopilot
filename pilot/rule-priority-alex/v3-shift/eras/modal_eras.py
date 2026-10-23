"""Alex's formula in the 40 copy heads (b40 from blockex/settings2.json):
  attention weight of a past example  ~  similarity (the model's own score)  x  sameness of conditions (w)
Done in log space: in the copy heads, from the test question onward, log(w) is added to the attention scores of the
PERIOD-1 examples (solved under the old rule). PERIOD-2 examples and everything else keep w = 1. w = 0 = full block
(= blockex2 'b40'). All other heads are untouched. Eager attention, per-head additive mask.
Settings: normal, w1 (formula with w = 1, must equal normal exactly), w0.3, w0.1, w0.03, w0.
In the N_same control the rule did not change, so the formula gives w = 1 (= normal); the w<1 numbers there show
the cost of applying the down-weighting where it should not be applied.
  modal run modal_eras.py [--n 4]   -> eras_<model>.jsonl (first row = checks)
"""
import json, math, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-eras14", image=image)
W = {"normal": None, "w1": 1.0, "w0.3": 0.3, "w0.1": 0.1, "w0.03": 0.03, "w0": 0.0}


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(model_name: str, heads: list, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    H = model.config.num_attention_heads
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    cur = {"w": None, "span": None}

    def mk(li):
        def pre(mod, args, kwargs):
            if cur["w"] is None or li not in by:
                return args, kwargs
            m = kwargs.get("attention_mask")
            if m is None:
                raise RuntimeError("no attention mask passed to the layer")
            b, _, q, k = m.shape
            if m.dtype == torch.bool:
                m = torch.zeros(m.shape, dtype=model.dtype, device=m.device).masked_fill(~m, torch.finfo(model.dtype).min)
            m = m.expand(b, H, q, k).clone()
            hs = torch.tensor(by[li], device=m.device)
            q0, a, e = cur["span"]
            if cur["w"] == 0:
                m[0, hs, q0:, a:e] = torch.finfo(m.dtype).min
            else:
                m[0, hs, q0:, a:e] = m[0, hs, q0:, a:e] + math.log(cur["w"])
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(mk(li), with_kwargs=True)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    checks = dict(max_w1_diff=0.0, spans={})
    out = []
    for n, it in enumerate(items):
        text = chat(it["prompt"])
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        a = text.find("Below are")
        if it["arm"] == "N_same":
            e = text.find("Below are 10 solved examples from PERIOD 2")
        else:
            e = text.find("IMPORTANT UPDATE")
        qc = text.find("Classify the TEST")
        assert 0 < a < e < qc, (a, e, qc)
        span = (pos(qc), pos(a), pos(e))
        if it["arm"] not in checks["spans"]:
            ids = enc["input_ids"][0]
            checks["spans"][it["arm"]] = dict(p1_start=tok.decode(ids[span[1]:span[1] + 6]), p1_end=tok.decode(ids[span[2] - 8:span[2]]),
                                              after_p1=tok.decode(ids[span[2]:span[2] + 4]), query=tok.decode(ids[span[0]:span[0] + 5]))
        lab = [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, w in W.items():
            cur.update(w=w, span=span)
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = torch.log_softmax(lg, -1)[lab].tolist()
        cur["w"] = None
        checks["max_w1_diff"] = max(checks["max_w1_diff"], max(abs(x - y) for x, y in zip(res["w1"], res["normal"])))
        pr = it["probes"][0]
        out.append(json.dumps(dict(set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"], key=pr["key"], old=pr["old"],
                                   scores={k: [round(x, 3) for x in v] for k, v in res.items()})))
        if (n + 1) % 20 == 0:
            print("eras", n + 1, "/", len(items), "max w1 diff", checks["max_w1_diff"], flush=True)
    yield json.dumps(dict(checks=checks))
    for line in out:
        yield line


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n: int = 0, tag: str = ""):
    heads = json.load(open(os.path.join(HERE, "settings2.json")))["b40"]
    items = json.load(open(os.path.join(HERE, "sets_eras.json")))
    if n:  # smoke test: first n items of each arm
        items = [x for arm in ("A60", "E_copy", "E_nocopy", "N_same") for x in [y for y in items if y["arm"] == arm][:n]]
    out = os.path.join(HERE, "eras_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, heads, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
