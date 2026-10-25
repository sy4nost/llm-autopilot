"""Mechanics test on a tiny random GPT-2 and a tokenizer trained on our own sentences (no downloads, no meaning)."""
import random, torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import obs_data, train_obs as T
from observer import Frozen, Observer

items = obs_data.build(1, 20)
bpe = ByteLevelBPETokenizer(); bpe.train_from_iterator([x["text"] + x["target"] + x["foil"] for x in items] * 5, vocab_size=400, min_frequency=1)
tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer); tok.pad_token = "!"
for n in obs_data.NAMES[:4]:
    print(n, tok.encode(" " + n))
torch.manual_seed(0)
m = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_layer=3, n_head=4, n_embd=32, n_positions=256), ).eval()
m.config._attn_implementation = "eager"
fz = Frozen(m); dev = "cpu"
print("check", T.check(fz, tok, items, dev))
tr = [x for x in items if x["tset"] == "A"]; te = [x for x in items if x["tset"] == "B"]
P_tr, P_te = T.precompute(fz, tok, tr, dev, bs=8), T.precompute(fz, tok, te, dev, bs=8)
plain = T.fill_plain_cands(fz, tok, T.plain_items([" The weather today is", " In the morning we", " She opened the door and"]), dev)
P_pl = T.precompute(fz, tok, plain, dev, bs=8)
nf = P_tr[0]["f"].shape[1]; print("features", nf)
obs = Observer(nf, 3, 4)
T.train_stage(fz, obs, P_tr, P_te, dev, 1, 1, 1e-3)
T.train_stage(fz, obs, P_tr, P_te, dev, 2, 1, 1e-3, plain=P_pl, plain_dev=P_pl)
r = T.evaluate(fz, obs, P_te, dev, 2, plain=P_pl, notes=True); print({k: r[k] for k in ("_alarm_auc", "_dev_score", "_plain")}, "notes", r["_notes"]["shape"])
print("shuffled", T.evaluate(fz, obs, P_te, dev, 2, shuffle=True)["_dev_score"])
const = torch.zeros(3, 4, requires_grad=True)
T.train_stage(fz, obs, P_tr, P_te, dev, 2, 1, 1e-2, const=const)
print("const", T.evaluate(fz, obs, P_te, dev, 2, const=const)["_dev_score"])
const2 = torch.zeros(2, 3, 4, requires_grad=True)
T.train_stage(fz, obs, P_tr, P_te, dev, 2, 1, 1e-2, const=const2, plain=P_pl, plain_dev=P_pl)
print("K2", T.evaluate(fz, obs, P_te, dev, 2, const=const2)["_dev_score"], "auc", T.evaluate(fz, obs, P_te, dev, 1)["_alarm_auc"])
print("R", T.check(fz, tok, T.no_history([x for x in te if x["cond"] in ("trap", "echo")]), dev))
# gains really reach the heads: zero gains must change the output
e = P_te[0]["enc"]; g0 = torch.zeros(e[0].shape[0], 3, 4)
print("gain effect", (T.run_gains(fz, e, g0, dev) - P_te[0]["logits"]).abs().max().item(),
      "unit gains same", (T.run_gains(fz, e, torch.ones_like(g0), dev) - P_te[0]["logits"]).abs().max().item())
w0 = obs.knobs.weight.detach().clone(); T.train_stage(fz, obs, P_tr, P_te, dev, 2, 1, 1e-2)
print("knobs moved", (obs.knobs.weight - w0).abs().max().item(), "const moved", const.abs().max().item())
