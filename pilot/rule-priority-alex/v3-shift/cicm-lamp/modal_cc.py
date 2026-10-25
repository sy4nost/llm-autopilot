"""cicm-lamp: Qwen2.5-14B on CICM dialogues. Per row: forced choice over the slot's 7 values (normal + heads-off passes for
our frozen top20 copy heads and layer-matched random head sets), residual at last prompt token (layers 24/32/40),
greedy 8-token answer. Heads are zeroed at the o_proj input from the first token of the final user message onward.
  modal run --detach modal_cc.py --tag smoke --split smoke ;  modal run modal_cc.py --tag smoke --get"""
import json, os, random
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "numpy")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-cicmlamp", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
LAYERS = (24, 32, 40)


def head_sets():
    top = [tuple(h) for h in json.load(open(os.path.join(HERE, "..", "lamp", "top20.json")))]
    rnd = random.Random(20261024); out = {"top20": top}
    for r in range(int(os.environ.get("NRAND", "6"))):
        s = []
        for l, _ in top:  # same per-layer counts as top20, heads not in top20
            while True:
                h = (l, rnd.randrange(40))
                if h not in top and h not in s:
                    s.append(h); break
        out[f"rand{r + 1}"] = s
    return out


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=7200)
def run(blob: str, sets_json: str, tag: str):
    import torch, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = json.loads(gzip.decompress(base64.b64decode(blob))); sets = {k: [tuple(h) for h in v] for k, v in json.loads(sets_json).items()}
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers; H = model.config.num_attention_heads; D = model.config.hidden_size // H
    cur = {"heads": None}

    def mk(li):
        def p(mod, args):
            hs = cur["heads"]
            if not hs or li not in hs:
                return None
            x = args[0].clone()
            for h in hs[li]:
                x[:, cur["t0"]:, h * D:(h + 1) * D] = 0
            return (x,) + args[1:]
        return p
    hooks = [layer.self_attn.o_proj.register_forward_pre_hook(mk(li)) for li, layer in enumerate(layers)]

    def by_layer(hl):
        d = {}
        for l, h in hl:
            d.setdefault(l, []).append(h)
        return d

    def cand_scores(pid, cands):
        # prompt pass (heads-off from t0 if set), then all candidates in one batch on a copy of the prompt cache
        import copy
        with torch.no_grad():
            o = model(input_ids=torch.tensor([pid]).cuda(), use_cache=True, output_hidden_states=cur.get("hid", False))
        hid = [o.hidden_states[L][0, -1].float().cpu().tolist() for L in LAYERS] if cur.get("hid") else None
        last = o.logits[0, -1].float().log_softmax(-1)
        toks = [tok(c, add_special_tokens=False)["input_ids"] for c in cands]
        m = max(len(t) for t in toks); res = [last[t[0]].item() for t in toks]
        if m > 1:
            kv = copy.deepcopy(o.past_key_values); kv.batch_repeat_interleave(len(toks))
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.tensor([t + [pad] * (m - len(t)) for t in toks]).cuda()
            am = torch.tensor([[1] * len(pid) + [1] * len(t) + [0] * (m - len(t)) for t in toks]).cuda()
            t0 = cur.get("t0"); cur["t0"] = 0 if (t0 or 0) <= len(pid) else t0  # cached positions are all after the question start
            with torch.no_grad():
                lg = model(input_ids=ids, attention_mask=am, past_key_values=kv, use_cache=True).logits.float().log_softmax(-1)
            cur["t0"] = t0
            for j, t in enumerate(toks):
                res[j] += sum(lg[j, i, t[i + 1]].item() for i in range(len(t) - 1))
        return [round(x, 4) for x in res], hid

    out = []; d = f"/cache/cicmlamp/{tag}"; os.makedirs(d, exist_ok=True)
    for n, r in enumerate(rows):
        msgs = r["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        head_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=False)
        assert text.startswith(head_text)
        pid = tok(text, add_special_tokens=False)["input_ids"]
        hid_ids = tok(head_text, add_special_tokens=False)["input_ids"]; t0 = len(hid_ids)
        assert pid[:t0] == hid_ids
        rec = dict(id=r["id"], cands=r["slot_values"], t0=t0, L=len(pid), ntok=[len(tok(c, add_special_tokens=False)["input_ids"]) for c in r["slot_values"]])
        cur.update(heads=None, hid=True)
        rec["final"], rec["hid"] = cand_scores(pid, r["slot_values"])
        cur["hid"] = False
        for k, hl in sets.items():
            cur.update(heads=by_layer(hl), t0=t0)
            rec["off_" + k], _ = cand_scores(pid, r["slot_values"])
        if n < 3:  # smoke gates: cached vs full recompute (heads on / top20 off), and hook active but zeroing nothing
            def full(hl):
                cur.update(heads=by_layer(hl) if hl else None, t0=t0); res = []
                for c in r["slot_values"]:
                    t = tok(c, add_special_tokens=False)["input_ids"]
                    with torch.no_grad():
                        lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float().log_softmax(-1)
                    res.append(round(sum(lg[i, t[i]].item() for i in range(len(t))), 4))
                return res
            rec["final_full"] = full(None); rec["off_full"] = full(sets["top20"])
            cur.update(heads=by_layer(sets["top20"]), t0=len(pid) + 50)
            rec["ident"], _ = cand_scores(pid, r["slot_values"])
        cur["heads"] = None
        with torch.no_grad():
            g = model.generate(torch.tensor([pid]).cuda(), max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
        rec["gen"] = tok.decode(g[0, len(pid):], skip_special_tokens=True)
        out.append(rec)
        if n % 50 == 0:
            print(n, len(pid), flush=True)
            open(f"{d}/out.json", "w").write(json.dumps(out)); cache.commit()
    open(f"{d}/out.json", "w").write(json.dumps(out)); open(f"{d}/DONE", "w").write("1"); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, part: int):
    import gzip, base64
    cache.reload()
    b = base64.b64encode(gzip.compress(open(f"/cache/cicmlamp/{tag}/out.json", "rb").read())).decode()
    done = os.path.exists(f"/cache/cicmlamp/{tag}/DONE")
    C = 900_000
    return json.dumps(dict(chunk=b[part * C:(part + 1) * C], more=(part + 1) * C < len(b), done=done))


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False, split: str = "all"):
    import gzip, base64
    if get:
        parts, k = [], 0
        while True:
            r = json.loads(fetch.remote(tag, k)); parts.append(r["chunk"]); k += 1
            if not r["more"]:
                break
        open(os.path.join(HERE, f"out_{tag}.json"), "wb").write(gzip.decompress(base64.b64decode("".join(parts))))
        if r["done"]:
            open(os.path.join(HERE, f"out_{tag}.DONE"), "w").write("1")
        print("fetched", tag, "done" if r["done"] else "PARTIAL"); return
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "cicm_natural_factorial_otherdist_l0.jsonl"))]
    import split_cc
    cal, conf = split_cc.split(rows)
    if split == "smoke":
        from collections import defaultdict
        g = defaultdict(list)
        for r in cal:
            g[str(r["factorial_cell"])].append(r)
        rows = [r for k in sorted(g) for r in g[k][:4]]
    else:
        rows = cal if split == "cal" else conf if split == "conf" else cal + conf
    if n:
        rows = rows[:n]
    sets = head_sets(); json.dump(sets, open(os.path.join(HERE, f"heads_{tag}.json"), "w"))
    blob = base64.b64encode(gzip.compress(json.dumps(rows).encode())).decode()
    print(len(rows), "rows; head sets", list(sets))
    print(run.remote(blob, json.dumps(sets), tag))
