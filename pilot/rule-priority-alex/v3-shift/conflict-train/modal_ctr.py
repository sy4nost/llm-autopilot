"""conflict-train runner (PLAN.md). Detector prompt: ctr_prompt.py. Model Qwen2.5-7B-Instruct, bf16, greedy, 48 new tokens.
  modal run modal_ctr.py --step eval --arm U --sets own,ext,wc [--n 30] --tag smoke  (untrained)
  modal run modal_ctr.py --step train --name L1                                     (LoRA on the train split, early stop on dev)
  modal run modal_ctr.py --step eval --arm L1 --sets own,ext,wc --tag main
  modal run modal_ctr.py --step get --tag main --arm U                              (download outputs)
Every reply is prefilled with ctr_prompt.PREFIX ("Conflict:"). p_yes = P(yes)/(P(yes)+P(no)) for the first generated token.
Outputs: out/<tag>_<arm>.json = {item id: {"text": prefix + generated line, "p_yes": float}}; train log: out/train_<name>.json."""
import json, os, sys, random
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ctr_prompt  # noqa: E402
from ctr_prompt import PREFIX  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate==1.14.0", "peft==0.21.1")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
         .add_local_python_source("ctr_prompt"))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-ctr", image=image)
D = "/cache/ctr"
NEW, BS, TOKBUDGET = 48, 16, 24000


def pack(obj):
    import gzip, base64
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def unpack(s):
    import gzip, base64
    return json.loads(gzip.decompress(base64.b64decode(s)))


def load(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, f"{D}/adapters/{adapter}")
    return tok, model.eval()


def prompt(tok, x):
    return tok.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=True) + PREFIX


def verdict_ids(tok):
    ids = {}
    for lab in ("yes", "no"):
        ids[lab] = sorted({tok.encode(w, add_special_tokens=False)[0] for w in (f" {lab}", f" {lab.capitalize()}", lab)})
    assert not set(ids["yes"]) & set(ids["no"]), ids
    return ids


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=5400)
def run_eval(blob: str, tag: str, arm: str):
    import torch
    items = unpack(blob); tok, model = load(None if arm == "U" else arm)
    texts = [prompt(tok, x) for x in items]; V = verdict_ids(tok)
    L = [len(tok(t, add_special_tokens=False)["input_ids"]) + NEW for t in texts]
    print(arm, "items", len(items), "max tokens", max(L), flush=True)
    order = sorted(range(len(items)), key=lambda i: L[i]); batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1 > BS or (len(cur) + 1) * L[i] > TOKBUDGET):
            batches.append(cur); cur = []
        cur.append(i)
    batches += [cur] if cur else []
    os.makedirs(D, exist_ok=True); out, done = {}, 0
    for bn, idx in enumerate(batches):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                               repetition_penalty=1.0, pad_token_id=tok.pad_token_id, return_dict_in_generate=True, output_scores=True)
        pr = torch.softmax(g.scores[0].float(), -1)
        py, pn = pr[:, V["yes"]].sum(-1), pr[:, V["no"]].sum(-1)
        for j, i in enumerate(idx):
            out[items[i]["id"]] = dict(text=PREFIX + tok.decode(g.sequences[j, enc.input_ids.shape[1]:], skip_special_tokens=True),
                                       p_yes=float(py[j] / (py[j] + pn[j] + 1e-12)), p_mass=float(py[j] + pn[j]))
        done += len(idx); print(arm, done, "/", len(items), flush=True)
        if bn % 10 == 0:
            open(f"{D}/{tag}_{arm}.json", "w").write(json.dumps(out)); cache.commit()
    open(f"{D}/{tag}_{arm}.json", "w").write(json.dumps(out)); open(f"{D}/{tag}_{arm}.DONE", "w").write("1"); cache.commit()
    return f"saved {len(out)}"


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=7200)
def train(blob_tr: str, blob_dev: str, name: str, epochs: int = 2, lr: float = 1e-4, rank: int = 16, seed: int = 20261025):
    """LoRA on attention+MLP projections; loss only on the target line; dev loss every half epoch; keep the best adapter."""
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed); random.seed(seed)
    tr, dev = unpack(blob_tr), unpack(blob_dev); tok, model = load(None)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    V = verdict_ids(tok)

    def enc(x):
        assert x["target"].startswith(PREFIX)
        pi = tok(prompt(tok, x), add_special_tokens=False)["input_ids"]
        ti = tok(x["target"][len(PREFIX):] + tok.eos_token, add_special_tokens=False)["input_ids"]
        assert ti[0] in V["yes"] + V["no"], tok.decode(ti[:2])
        return pi + ti, [-100] * len(pi) + ti

    E_tr, E_dev = [enc(x) for x in tr], [enc(x) for x in dev]

    def batches(E, shuffle):
        ix = list(range(len(E)))
        if shuffle:
            random.shuffle(ix)
        out, cur = [], []
        for i in ix:
            if cur and (len(cur) + 1) * max(len(E[j][0]) for j in cur + [i]) > 8000:
                out.append(cur); cur = []
            cur.append(i)
        return out + ([cur] if cur else [])

    def collate(E, idx):
        m = max(len(E[i][0]) for i in idx)
        ids = torch.full((len(idx), m), tok.pad_token_id); lab = torch.full((len(idx), m), -100); att = torch.zeros((len(idx), m), dtype=torch.long)
        for k, i in enumerate(idx):
            a, b = E[i]; ids[k, :len(a)] = torch.tensor(a); lab[k, :len(b)] = torch.tensor(b); att[k, :len(a)] = 1
        return ids.cuda(), lab.cuda(), att.cuda()

    def dev_loss():
        """Early-stopping signal: two-way (yes vs no) loss on the verdict token only, plus verdict accuracy and full-target loss."""
        model.eval(); s = n = vl = acc = 0
        with torch.no_grad():
            for idx in batches(E_dev, False):
                ids, lab, att = collate(E_dev, idx); o = model(input_ids=ids, attention_mask=att, labels=lab)
                s += o.loss.item() * len(idx); n += len(idx)
                for k, i in enumerate(idx):
                    pos = next(p for p, t in enumerate(E_dev[i][1]) if t != -100)          # first target token = the verdict
                    lp = torch.log_softmax(o.logits[k, pos - 1].float(), -1)
                    ly, ln = torch.logsumexp(lp[V["yes"]], 0), torch.logsumexp(lp[V["no"]], 0)
                    gold_yes = E_dev[i][0][pos] in V["yes"]
                    vl += -(ly if gold_yes else ln).item() + torch.logsumexp(torch.stack([ly, ln]), 0).item()
                    acc += int((ly > ln).item() == gold_yes)
        model.train(); return vl / n, acc / n, s / n

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    plan = [batches(E_tr, True) for _ in range(epochs)]     # all epochs' shuffled batches, so the schedule knows the true step count
    steps_total = sum(len(B) for B in plan); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))
    log, best, step = dict(dev=[], train=[]), 1e9, 0
    model.train(); log["dev"].append((0, dev_loss())); print("dev loss start", log["dev"][-1], flush=True)
    for ep in range(epochs):
        B = plan[ep]; half = len(B) // 2
        for bi, idx in enumerate(B):
            ids, lab, att = collate(E_tr, idx)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            log["train"].append((step, loss.item()))
            if step % 20 == 0:
                print("ep", ep, "step", step, "loss", round(loss.item(), 4), flush=True)
            if bi + 1 in (half, len(B)):
                dv = dev_loss(); log["dev"].append((step,) + dv); dl = dv[0]
                print("dev verdict loss / acc / target loss", step, [round(v, 4) for v in dv], flush=True)
                if dl < best:
                    best = dl; model.save_pretrained(f"{D}/adapters/{name}"); log["best_step"] = step; cache.commit()
    os.makedirs(D, exist_ok=True); open(f"{D}/train_{name}.json", "w").write(json.dumps(log)); cache.commit()
    return f"best dev verdict loss {best:.4f} at step {log.get('best_step')}"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(fn: str):
    cache.reload()
    return json.dumps(dict(out=open(f"{D}/{fn}").read() if os.path.exists(f"{D}/{fn}") else None,
                           done=os.path.exists(f"{D}/{fn.replace('.json', '.DONE')}")))


def items_for(sets, n):
    xs = []
    if "own" in sets:
        xs += [json.loads(l) for l in open(os.path.join(HERE, "own_items.jsonl"))]
        xs = [dict(x, id=f"own{x['id']}") for x in xs if x["split"] in ("test", "dev")]
    if "ext" in sets:
        xs += [json.loads(l) for l in open(os.path.join(HERE, "ext_items.jsonl"))]
    if "wc" in sets and os.path.exists(os.path.join(HERE, "wc_items.jsonl")):
        xs += [json.loads(l) for l in open(os.path.join(HERE, "wc_items.jsonl")) if json.loads(l).get("split") == "test"]
    if n:
        xs = random.Random(2).sample(xs, n)
    return [dict(id=x["id"], messages=x["messages"]) for x in xs]


@app.local_entrypoint()
def main(step: str = "eval", arm: str = "U", sets: str = "own,ext,wc", n: int = 0, tag: str = "smoke", name: str = "L1", epochs: int = 2):
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    if step == "get":
        for fn in (f"{tag}_{arm}.json", f"train_{arm}.json"):
            r = json.loads(fetch.remote(fn))
            if r["out"]:
                open(os.path.join(HERE, "out", fn), "w").write(r["out"]); print(fn, "fetched", "done" if r["done"] else "")
        return
    if step == "eval":
        xs = [dict(id=x["id"], chat=ctr_prompt.chat(x["messages"])) for x in items_for(sets.split(","), n)]
        print("items", len(xs))
        print(run_eval.remote(pack(xs), tag, arm)); return
    if step == "train":
        own = [json.loads(l) for l in open(os.path.join(HERE, "own_items.jsonl"))]
        extra = [json.loads(l) for l in open(os.path.join(HERE, "wc_items.jsonl"))] if os.path.exists(os.path.join(HERE, "wc_items.jsonl")) else []
        tr = [x for x in own + extra if x["split"] == "train"]; dev = [x for x in own + extra if x["split"] == "dev"]
        if n:
            tr, dev = tr[:n], dev[:max(8, n // 4)]
        f = lambda X: [dict(chat=ctr_prompt.chat(x["messages"]), target=x["target"]) for x in X]
        print("train", len(tr), "dev", len(dev)); print(train.remote(pack(f(tr)), pack(f(dev)), name, epochs))
