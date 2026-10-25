"""Train and test the observer floor on GPT-2 small (see observer.py, obs_data.py, PLAN.md).
Steps (each prints a table; results go to a JSON):
  check   does the habit trap exist? accuracy of the frozen model on clean / trap / echo / control, both template sets
  stage1  observer only WATCHES: learns to predict 'the frozen answer will be wrong' (alarm); no knobs
  stage2  observer may ACT: alarm + per-head gains, trained on the final answer, with a cost per change and a
          'leave it alone' penalty on items the frozen model already gets right and on plain text
  Controls: K = constant gains (same for every input, like a fixed retuning); K2 = two constant gain sets switched by a
  trivial input flag (an earlier sentence ended with one of the candidate names: trap AND echo); S = observer gains given to
  a random other item of the whole test set; R = the same final sentences with the history removed (paired upper bound)."""
import json, math, random
import torch
import torch.nn.functional as F
from observer import Frozen, Observer, features, last_index, N_REGION


def encode(tok, items, device):
    enc = tok([x["text"] for x in items], return_offsets_mapping=True, padding=True, return_tensors="pt", add_special_tokens=False)
    ids, att, off = enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
    B, T = ids.shape; regions = torch.zeros(B, N_REGION, T); cands = torch.zeros(B, 2, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long); foil = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(items):
        c = x.get("cand_ids") or [tok.encode(s, add_special_tokens=False)[0] for s in x["cand"]]
        cands[i] = torch.tensor(c)
        if "target" in x:
            tgt[i] = tok.encode(x["target"], add_special_tokens=False)[0]; foil[i] = tok.encode(x["foil"], add_special_tokens=False)[0]
        n = int(att[i].sum()); hc = x.get("hist_chars", 0)
        for t in range(n):
            s, e = off[i, t].tolist(); in_hist = e <= hc
            regions[i, 0 if in_hist else 1, t] = 1
            if t == 0:
                regions[i, 2, t] = 1
            if not in_hist and ids[i, t] == c[0]:
                regions[i, 3, t] = 1
            if not in_hist and ids[i, t] == c[1]:
                regions[i, 4, t] = 1
            if in_hist and t + 1 < n and tok.decode([int(ids[i, t + 1])]).strip() == "." and int(ids[i, t]) in c:
                regions[i, 5, t] = 1                                        # the name that ended a history sentence
    return ids.to(device), att.to(device), regions.to(device), cands.to(device), tgt.to(device), foil.to(device)


def margin(logits, a, b):
    ar = torch.arange(logits.shape[0], device=logits.device)
    return logits[ar, a] - logits[ar, b]


def precompute(fz, tok, items, device, bs=32):
    """Pass 1 for every item: features, frozen logits at the answer position, encoded tensors. Kept on CPU."""
    out = []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]; e = encode(tok, chunk, device)
        f, lg = features(fz, e[0], e[1], e[2], e[3])
        out.append(dict(items=chunk, enc=[t.cpu() for t in e], f=f.cpu(), logits=lg.float().cpu()))
    return out


def run_gains(fz, enc, gains, device):
    ids, att = enc[0].to(device), enc[1].to(device)
    o = fz.forward(ids, att, gains=gains)
    return o.logits[torch.arange(ids.shape[0], device=device), last_index(att)]


def auc(s, y):
    pos = [a for a, b in zip(s, y) if b]; neg = [a for a, b in zip(s, y) if not b]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def table(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault((r["tset"], r["cond"]), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        out["/".join(g)] = {k: round(sum(r[k] for r in rs) / len(rs), 3) for k in keys}
        out["/".join(g)]["n"] = len(rs)
    return out


def check(fz, tok, items, device):
    P = precompute(fz, tok, items, device)
    rows = []
    for b in P:
        m = margin(b["logits"], b["enc"][4], b["enc"][5])
        for x, v in zip(b["items"], m.tolist()):
            rows.append(dict(tset=x["tset"], cond=x["cond"], acc=float(v > 0)))
    return table(rows, ["acc"])


def const_gains(const, enc, device):
    """K: one gain vector for every input ([L,H]). K2: two vectors chosen by a trivial input flag ([2,L,H]):
    flag = the history contains a sentence that ended with one of the two candidate names (true for trap AND echo)."""
    B = enc[0].shape[0]
    if const.dim() == 2:
        return (1 + torch.tanh(const))[None].expand(B, -1, -1)
    flag = (enc[2][:, 5].sum(-1) > 0).long().to(device)
    return 1 + torch.tanh(const[flag])


def gains_of(obs, b, device, const=None):
    if const is not None:
        return const_gains(const, b["enc"], device)
    return obs(b["f"].to(device))[1]


def train_stage(fz, obs, P_tr, P_dev, device, stage, epochs, lr, lam_cost=0.05, beta_keep=1.0, const=None, plain=None,
                plain_dev=None, log=print):
    """stage 1: alarm only (BCE on 'frozen answer wrong'). stage 2: gains (answer loss + cost + keep penalty).
    Task and plain batches are mixed in random order every epoch."""
    params = list(obs.parameters()) if const is None else [const]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    best, best_state = -1e9, None
    for ep in range(epochs):
        obs.train(); order = P_tr + ([] if stage == 1 else (plain or [])); random.shuffle(order)
        for b in order:
            f = b["f"].to(device); lg1 = b["logits"].to(device); enc = b["enc"]
            is_plain = "target" not in b["items"][0]
            if stage == 1:
                a, _ = obs(f); wrong = (margin(lg1, enc[4].to(device), enc[5].to(device)) <= 0).float()
                loss = F.binary_cross_entropy(a, wrong)
            else:
                g = gains_of(obs, b, device, const)
                lg2 = run_gains(fz, enc, g, device)
                keep = F.kl_div(F.log_softmax(lg2, -1), F.log_softmax(lg1, -1), log_target=True, reduction="none").sum(-1)
                cost = (g - 1).abs().mean((1, 2))
                if is_plain:
                    loss = (beta_keep * keep + lam_cost * cost).mean()
                else:
                    t, fo = enc[4].to(device), enc[5].to(device)
                    two = torch.stack([lg2[torch.arange(len(t)), t], lg2[torch.arange(len(t)), fo]], 1)
                    ce = F.cross_entropy(two, torch.zeros(len(t), dtype=torch.long, device=device), reduction="none")
                    right1 = (margin(lg1, t, fo) > 0).float()
                    loss = (ce + beta_keep * right1 * keep + lam_cost * cost).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        r = evaluate(fz, obs, P_dev, device, stage, const, plain=plain_dev if stage == 2 else None)
        score = r["_dev_score"]
        if stage == 2 and "_plain" in r:                       # breaking plain text costs the dev score
            score -= 2 * max(0.0, 0.98 - r["_plain"]["top1_same"])
        log(f"stage {stage} epoch {ep} dev score {score:.3f}")
        if score > best:
            best = score; best_state = {k: v.detach().clone() for k, v in (obs.state_dict().items() if const is None else {"c": const}.items())}
    if const is None:
        obs.load_state_dict(best_state)
    else:
        const.data.copy_(best_state["c"])
    return best


@torch.no_grad()
def evaluate(fz, obs, P, device, stage, const=None, shuffle=False, plain=None, seed=0, notes=False):
    """shuffle=True: control S, every item gets the gains the observer chose for a random OTHER item of the whole set."""
    obs.eval(); rows = []; nb = []
    acting = stage == 2 or const is not None
    G = [gains_of(obs, b, device, const) for b in P] if acting else None
    if acting and shuffle:
        allg = torch.cat(G); gen = torch.Generator().manual_seed(seed)
        allg = allg[torch.randperm(allg.shape[0], generator=gen).to(device)]
        G = list(torch.split(allg, [g.shape[0] for g in G]))
    for k, b in enumerate(P):
        f = b["f"].to(device); lg1 = b["logits"].to(device); enc = b["enc"]
        t, fo = enc[4].to(device), enc[5].to(device)
        m1 = margin(lg1, t, fo); a, _ = obs(f)
        if notes:
            nb.append(obs.note(f).half().cpu())
        if acting:
            g = G[k]; m2 = margin(run_gains(fz, enc, g, device), t, fo); ch = (g - 1).abs().mean((1, 2))
        else:
            m2 = m1; ch = torch.zeros_like(m1)
        for x, u, v, al, c in zip(b["items"], m1.tolist(), m2.tolist(), a.tolist(), ch.tolist()):
            rows.append(dict(tset=x["tset"], cond=x["cond"], acc_before=float(u > 0), acc_after=float(v > 0), alarm=al,
                             wrong_before=float(u <= 0), change=c))
    res = table(rows, ["acc_before", "acc_after", "alarm", "change"])
    res["_alarm_auc"] = {}
    for ts in sorted({r["tset"] for r in rows}):
        for cond in ["all"] + sorted({r["cond"] for r in rows}):
            rs = [r for r in rows if r["tset"] == ts and (cond == "all" or r["cond"] == cond)]
            res["_alarm_auc"][f"{ts}/{cond}"] = auc([r["alarm"] for r in rs], [r["wrong_before"] for r in rs])
    res["_rows"] = [[r["tset"], r["cond"], r["acc_before"], r["acc_after"], round(r["alarm"], 4)] for r in rows]
    if stage == 1:
        res["_dev_score"] = auc([r["alarm"] for r in rows], [r["wrong_before"] for r in rows])
    else:   # trap accuracy gained, minus twice any accuracy lost on the other conditions
        tr = [r for r in rows if r["cond"] == "trap"]; ok = [r for r in rows if r["cond"] != "trap"]
        res["_dev_score"] = (sum(r["acc_after"] - r["acc_before"] for r in tr) / max(1, len(tr))
                             - 2 * max(0.0, sum(r["acc_before"] - r["acc_after"] for r in ok) / max(1, len(ok))))
    if notes:   # the observer's notebook: one row of hidden numbers per item, same order as _rows (float16, base64)
        import base64
        allnb = torch.cat(nb).numpy()
        res["_notes"] = dict(shape=list(allnb.shape), dtype="float16", b64=base64.b64encode(allnb.tobytes()).decode())
    if plain:
        agree, kl = [], []
        for b in plain:
            lg1 = b["logits"].to(device); lg2 = run_gains(fz, b["enc"], gains_of(obs, b, device, const), device)
            agree += (lg2.argmax(-1) == lg1.argmax(-1)).float().tolist()
            kl += F.kl_div(F.log_softmax(lg2, -1), F.log_softmax(lg1, -1), log_target=True, reduction="none").sum(-1).tolist()
        res["_plain"] = dict(top1_same=sum(agree) / len(agree), kl=sum(kl) / len(kl), n=len(agree))
    return res


def no_history(items):
    """R: the same final sentences with the history removed (paired upper bound)."""
    return [dict(x, text=x["final"], hist_chars=0) for x in items]


def plain_items(texts):
    """Plain text: candidates are filled in later from the frozen model's own top-2 guesses (no labels)."""
    return [dict(text=t, hist_chars=0, cand=None, tset="plain", cond="plain") for t in texts]


def fill_plain_cands(fz, tok, items, device):
    for k in range(0, len(items), 32):
        chunk = items[k:k + 32]
        enc = tok([x["text"] for x in chunk], padding=True, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            o = fz.forward(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        li = last_index(enc["attention_mask"].to(device))
        top = o.logits[torch.arange(len(chunk), device=device), li].topk(2, -1).indices.tolist()
        for x, tt in zip(chunk, top):
            x["cand"] = [tok.decode([tt[0]]), tok.decode([tt[1]])]; x["cand_ids"] = tt
    return items
