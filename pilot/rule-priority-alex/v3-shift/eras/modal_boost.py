"""Make the NEW rule louder (mirror of Alex's formula): from the test question onward, add log(k) to the attention
scores on the update line ('IMPORTANT UPDATE ...' paragraph), optionally together with the formula's log(w) on the
PERIOD-1 examples in the 40 copy heads (b40). Prompts: sets_eras.json arms A60 / E_copy / E_nocopy, plus prompt-only
variants A60_rep / E_nocopy_rep where the update paragraph is repeated right before the test question.
Head sets for the boost: b40 (copy heads), late = all heads of layers 29-47, mid = layers 14-28, all = every layer.
Settings: normal, up1_late (k=1, must equal normal), up4_b40, up4_late, up8_late, up4_mid, up4_all, up16_all, w0.1 (old examples only),
w0.1+up4_late.
  modal run modal_boost.py [--n 2]   -> boost_<model>.jsonl (first row = checks)
"""
import json, math, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-boost14", image=image)
# setting -> list of (head set, span, multiplier)
SETTINGS = {"normal": [], "up1_late": [("late", "upd", 1.0)], "up4_b40": [("b40", "upd", 4.0)], "up4_late": [("late", "upd", 4.0)],
            "up8_late": [("late", "upd", 8.0)], "up4_mid": [("mid", "upd", 4.0)], "up4_all": [("all", "upd", 4.0)],
            "up16_all": [("all", "upd", 16.0)], "w0.1": [("b40", "p1", 0.1)], "w0.1+up4_late": [("b40", "p1", 0.1), ("late", "upd", 4.0)]}


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
                a, e = cur["spans"][span]
                hs = torch.tensor(hs, device=m.device)
                m[0, hs, q0:, a:e] = m[0, hs, q0:, a:e] + math.log(w)
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(mk(li), with_kwargs=True)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    checks = dict(max_up1_diff=0.0, spans={})
    out = []
    for n, it in enumerate(items):
        text = chat(it["prompt"])
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        a = text.find("Below are")
        u = text.find("IMPORTANT UPDATE")
        ue = text.find("\n\n", u)
        qc = text.find("Classify the TEST")
        assert 0 < a < u < ue < qc, (a, u, ue, qc)
        spans = dict(q=pos(qc), p1=(pos(a), pos(u)), upd=(pos(u), pos(ue)))
        if it["arm"] not in checks["spans"]:
            ids0 = enc["input_ids"][0]
            checks["spans"][it["arm"]] = dict(upd=tok.decode(ids0[spans["upd"][0]:spans["upd"][1]]),
                                              p1_end=tok.decode(ids0[spans["p1"][1] - 6:spans["p1"][1]]), query=tok.decode(ids0[spans["q"]:spans["q"] + 5]))
        lab = [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, mods in SETTINGS.items():
            cur.update(mods=mods, spans=spans)
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = torch.log_softmax(lg, -1)[lab].tolist()
        cur["mods"] = []
        checks["max_up1_diff"] = max(checks["max_up1_diff"], max(abs(x - y) for x, y in zip(res["up1_late"], res["normal"])))
        pr = it["probes"][0]
        out.append(json.dumps(dict(set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"], key=pr["key"], old=pr["old"],
                                   scores={k: [round(x, 3) for x in v] for k, v in res.items()})))
        if (n + 1) % 20 == 0:
            print("boost", n + 1, "/", len(items), "max up1 diff", checks["max_up1_diff"], flush=True)
    yield json.dumps(dict(checks=checks))
    for line in out:
        yield line


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n: int = 0, tag: str = ""):
    b40 = json.load(open(os.path.join(HERE, "settings2.json")))["b40"]
    base = [x for x in json.load(open(os.path.join(HERE, "sets_eras.json"))) if x["arm"] in ("A60", "E_copy", "E_nocopy")]
    items = list(base)
    for x in base:
        if x["arm"] in ("A60", "E_nocopy"):
            p = x["prompt"]
            u = p.find("IMPORTANT UPDATE")
            note = p[u:p.find("\n\n", u) + 2]
            items.append(dict(x, arm=x["arm"] + "_rep", prompt=p.replace("Classify the TEST case below", "REMINDER: " + note + "Classify the TEST case below")))
    if n:
        items = [y for arm in ("A60", "E_copy", "E_nocopy", "A60_rep", "E_nocopy_rep") for y in [z for z in items if z["arm"] == arm][:n]]
    out = os.path.join(HERE, "boost_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, b40, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
