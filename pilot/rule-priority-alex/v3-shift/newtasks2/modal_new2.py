"""Frozen-settings run on the unfamiliar task types (run 2, newtasks2/PREREG.md; identical to newtasks/modal_new.py except names). Same hook as eras/modal_boost.py.
Settings: normal; w1 (check: must equal normal); w0 (PERIOD-1 / old examples silenced in the 40 copy heads b40, from the
test question onward); up1_all (check: must equal normal); up16_all (update note x16 louder in every head, from the question
onward); up4_all (secondary, reported only). Arms without an update note: up* settings are a no-op; F0 has no examples: w* no-op.
  modal run modal_new.py [--n 2] [--sets sets_new2.json]   -> new_<model>.jsonl (first row = checks)
"""
import json, math, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-new2-14", image=image)
# setting -> list of (head set, span, multiplier)
SETTINGS = {"normal": [], "w1": [("b40", "p1", 1.0)], "w0": [("b40", "p1", 0.0)], "up1_all": [("all", "upd", 1.0)],
            "up16_all": [("all", "upd", 16.0)], "up4_all": [("all", "upd", 4.0)]}


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(model_name: str, b40: list, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    H, NL = model.config.num_attention_heads, model.config.num_hidden_layers
    sets = {"b40": {}, "late": {l: list(range(H)) for l in range(29, NL)}, "mid": {l: list(range(H)) for l in range(14, 29)},
            "all": {l: list(range(H)) for l in range(NL)}}
    for l, h in b40:
        sets["b40"].setdefault(l, []).append(h)
    cur = {"mods": [], "spans": None}

    def mk(li):
        def pre(mod, args, kwargs):
            mods = [(sets[s][li], span, w) for s, span, w in cur["mods"] if li in sets[s]]
            if not mods:
                return args, kwargs
            m = kwargs.get("attention_mask")
            if m is None:
                raise RuntimeError("no attention mask passed to the layer")
            b, _, q, k = m.shape
            if m.dtype == torch.bool:
                m = torch.zeros(m.shape, dtype=model.dtype, device=m.device).masked_fill(~m, torch.finfo(model.dtype).min)
            m = m.expand(b, H, q, k).clone()
            q0 = cur["spans"]["q"]
            for hs, span, w in mods:
                if cur["spans"].get(span) is None:
                    continue
                a, e = cur["spans"][span]
                hs = torch.tensor(hs, device=m.device)
                if w > 0:
                    m[0, hs, q0:, a:e] = m[0, hs, q0:, a:e] + math.log(w)
                else:   # as in eras/modal_eras.py w0
                    m[0, hs, q0:, a:e] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(mk(li), with_kwargs=True)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    checks = dict(max_up1_diff=0.0, max_w1_diff=0.0, spans={})
    out = []
    for n, it in enumerate(items):
        text = chat(it["prompt"])
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        qc = text.find("Answer for the TEST")
        a = text.find("Below are")
        u = text.find("IMPORTANT UPDATE")
        a2 = text.find("Below are", a + 1) if a >= 0 else -1
        if a < 0:
            p1 = None
        else:
            ends = [x for x in (u, a2, qc) if x > a]
            p1 = (pos(a), pos(min(ends)))
        upd = None
        if u >= 0 and "reviewed and stays the same" not in text[u:text.find("\n\n", u)]:
            upd = (pos(u), pos(text.find("\n\n", u)))
        elif u >= 0:
            upd = (pos(u), pos(text.find("\n\n", u)))   # review note: boosted too (it is the control for a note that changes nothing)
        assert qc > 0 and (p1 is None or p1[0] < p1[1] <= pos(qc)), (a, u, qc)
        spans = dict(q=pos(qc), p1=p1, upd=upd)
        key_arm = it["task"] + ":" + it["arm"]
        if key_arm not in checks["spans"]:
            ids0 = enc["input_ids"][0]
            checks["spans"][key_arm] = dict(upd=None if upd is None else tok.decode(ids0[upd[0]:upd[1]]),
                                            p1_start=None if p1 is None else tok.decode(ids0[p1[0]:p1[0] + 4]),
                                            p1_end=None if p1 is None else tok.decode(ids0[p1[1] - 6:p1[1]]),
                                            query=tok.decode(ids0[spans["q"]:spans["q"] + 5]))
        lab = [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
        assert len(set(lab)) == len(lab), ("label first tokens collide", it["labels"])
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, mods in SETTINGS.items():
            cur.update(mods=mods, spans=spans)
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = torch.log_softmax(lg, -1)[lab].tolist()
        cur["mods"] = []
        checks["max_up1_diff"] = max(checks["max_up1_diff"], max(abs(x - y) for x, y in zip(res["up1_all"], res["normal"])))
        checks["max_w1_diff"] = max(checks["max_w1_diff"], max(abs(x - y) for x, y in zip(res["w1"], res["normal"])))
        out.append(json.dumps(dict(task=it["task"], set=it["set"], arm=it["arm"], probe=it["probe"], labels=it["labels"], key=it["key"], old=it["old"],
                                   scores={k: [round(x, 3) for x in v] for k, v in res.items()})))
        if (n + 1) % 20 == 0:
            print("new", n + 1, "/", len(items), "max up1/w1 diff", checks["max_up1_diff"], checks["max_w1_diff"], flush=True)
    yield json.dumps(dict(checks=checks))
    for line in out:
        yield line


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n: int = 0, sets: str = "sets_new2.json", tag: str = ""):
    b40 = json.load(open(os.path.join(HERE, "..", "eras", "settings2.json")))["b40"]
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    if n:
        seen, keep = {}, []
        for x in items:
            k = (x["task"], x["arm"])
            if seen.get(k, 0) < n:
                seen[k] = seen.get(k, 0) + 1; keep.append(x)
        items = keep
    out = os.path.join(HERE, "new2_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, b40, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
