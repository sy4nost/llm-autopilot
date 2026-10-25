"""Run the observer-floor pilot on Modal (GPT-2 small, frozen).
  modal run modal_obs.py --step check            # does the habit trap exist? (cents)
  modal run modal_obs.py --step train            # stage 1 (watch), stage 2 (act), controls K and S, test tables
Results: results_<step>.json next to this file."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "datasets==4.4.1")
         .add_local_python_source("observer", "obs_data", "train_obs"))
app = modal.App("silo-observer", image=image)
SEED_TR, SEED_DEV, SEED_TE, SEED_SEL = 20261025, 20261026, 20261027, 20261100


@app.function(gpu="T4", timeout=3600)
def run(step: str, b_templates: str = "", epochs1: int = 5, epochs2: int = 6, lam_cost: float = 0.05):
    import random, torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import obs_data, train_obs as T
    from observer import Frozen, Observer
    random.seed(0); torch.manual_seed(0); dev = "cuda"
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token; tok.padding_side = "right"
    for n in obs_data.NAMES + obs_data.PLACES + obs_data.OBJECTS:
        assert len(tok.encode(" " + n)) == 1, n
    for x in obs_data.build(1, 5):                  # every candidate name is found as its own token in the text
        ids = tok.encode(x["text"]); assert all(tok.encode(c)[0] in ids for c in x["cand"]), x["text"]
    fz = Frozen(GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev))
    if step == "select":                            # pick held-out wordings where the habit exists (own seed, not the test seed)
        rows = []
        for k, t in enumerate(obs_data.B_CANDIDATES):
            obs_data.TEMPLATES["B"] = [t]
            its = obs_data.build(SEED_SEL + k, 150, tsets=("B",))
            c = T.check(fz, tok, its, dev); rr = T.check(fz, tok, T.no_history([x for x in its if x["cond"] == "trap"]), dev)
            row = dict(k=k, template=t, **{cc: c["B/" + cc]["acc"] for cc in ("clean", "trap", "echo", "control")},
                       no_hist=rr["B/trap"]["acc"])
            row["drop"] = round(row["no_hist"] - row["trap"], 3)
            row["ok"] = row["clean"] >= .85 and row["echo"] >= .85 and row["no_hist"] >= .85 and row["drop"] >= .10
            rows.append(row); print(row, flush=True)
        keep = sorted([r for r in rows if r["ok"]], key=lambda r: -r["drop"])[:3]
        return dict(candidates=rows, chosen=[r["template"] for r in keep])
    if b_templates:                                 # wordings chosen by step 'select' (file results_select.json)
        obs_data.TEMPLATES["B"] = json.loads(b_templates)
    test = obs_data.build(SEED_TE, 300)
    R = T.check(fz, tok, T.no_history([x for x in test if x["cond"] in ("trap", "echo")]), dev)
    if step == "check":
        return dict(check=T.check(fz, tok, test, dev), R_no_history=R)
    from datasets import load_dataset
    wt = [l.strip() for l in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(l.split()) > 12]
    r = random.Random(1); r.shuffle(wt)
    cut = lambda t: " " + " ".join(t.split()[: r.randint(8, 30)])
    pl = {k: T.fill_plain_cands(fz, tok, T.plain_items([cut(t) for t in v]), dev)
          for k, v in dict(tr=wt[:400], dv=wt[400:550], te=wt[550:850]).items()}
    tr = obs_data.build(SEED_TR, 600, tsets=("A",)); dv = obs_data.build(SEED_DEV, 100, tsets=("A",))
    P = {k: T.precompute(fz, tok, v, dev) for k, v in dict(tr=tr, dv=dv, te=test).items()}
    PL = {k: T.precompute(fz, tok, v, dev) for k, v in pl.items()}
    nf = P["tr"][0]["f"].shape[1]; log = []
    say = lambda s: (print(s, flush=True), log.append(s))
    out = dict(n_features=nf, before=T.check(fz, tok, test, dev), R_no_history=R)
    obs = Observer(nf, fz.L, fz.H).to(dev)
    T.train_stage(fz, obs, P["tr"], P["dv"], dev, 1, epochs1, 1e-3, log=say)
    out["stage1_watch"] = T.evaluate(fz, obs, P["te"], dev, 1, notes=True)
    T.train_stage(fz, obs, P["tr"], P["dv"], dev, 2, epochs2, 3e-4, lam_cost=lam_cost, plain=PL["tr"], plain_dev=PL["dv"], log=say)
    out["stage2_act"] = T.evaluate(fz, obs, P["te"], dev, 2, plain=PL["te"], notes=True)
    out["control_S_shuffled"] = T.evaluate(fz, obs, P["te"], dev, 2, shuffle=True, plain=PL["te"])
    for name, shape in (("K_constant", (fz.L, fz.H)), ("K2_flag", (2, fz.L, fz.H))):
        const = torch.zeros(*shape, device=dev, requires_grad=True)
        T.train_stage(fz, obs, P["tr"], P["dv"], dev, 2, epochs2, 1e-2, lam_cost=lam_cost, const=const, plain=PL["tr"],
                      plain_dev=PL["dv"], log=say)
        out["control_" + name] = T.evaluate(fz, obs, P["te"], dev, 2, const=const, plain=PL["te"])
        out["control_" + name + "_gains"] = (1 + torch.tanh(const)).tolist()
    out["log"] = log
    return out


@app.local_entrypoint()
def main(step: str = "check", use_selected: bool = False):
    bt = json.dumps(json.load(open(os.path.join(HERE, "results_select.json")))["chosen"]) if use_selected else ""
    res = run.remote(step, bt)
    if bt:
        res["b_templates"] = json.loads(bt)
    open(os.path.join(HERE, f"results_{step}.json"), "w").write(json.dumps(res, indent=1, default=float))
    short = {k: ({kk: vv for kk, vv in v.items() if kk not in ("_rows", "_notes")} if isinstance(v, dict) else v) for k, v in res.items()
             if k != "log" and not k.endswith("_gains")}
    print(json.dumps(short, indent=1, default=float)[:8000])
