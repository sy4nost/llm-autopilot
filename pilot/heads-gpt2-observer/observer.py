"""The 'observer floor' for GPT-2 (Alex's idea): a small network that watches every head of a frozen model and can turn
each head's volume down or up before the model answers.

Pass 1 (watch): the frozen model reads the text; hooks record, at the answer position,
  1. where each head looks: share of attention on the history, on the current sentence, on the first token,
     on each of the two candidate names, and on the earlier answer token(s) in the history (the copy target);
  2. (part of 1) the copy signal = attention on those earlier answer tokens;
  3. the leader by floor: margin between the two candidate names after every floor (logit lens);
  4. the tug of war: what each head writes directly toward candidate 1 vs candidate 2.
  Nothing here uses the right answer: the candidates are just the two names of the final sentence, in order.
Pass 2 (act): the observer outputs an alarm a in [0,1] and per-head gains g = 1 + a * tanh(w) in [0,2];
  the frozen model reads the text again with every head's output multiplied by its gain.
Only the observer's weights are trained; GPT-2 stays frozen."""
import torch
import torch.nn as nn

N_REGION = 6   # history, current sentence, first token, candidate 1, candidate 2, earlier answers in the history


class Frozen:
    """Wraps a Hugging Face GPT-2: hooks to read each head's output (z) and to scale it by a gain."""

    def __init__(self, model):
        self.m = model.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        cfg = model.config; self.L, self.H = cfg.n_layer, cfg.n_head; self.D = cfg.n_embd // cfg.n_head
        self.gains = None; self.z = {}; self.pre_lnf = None
        self.m.transformer.ln_f.register_forward_pre_hook(lambda mod, a: setattr(self, "pre_lnf", a[0]))
        for l, blk in enumerate(self.m.transformer.h):
            blk.attn.c_proj.register_forward_pre_hook(self._hook(l))

    def _hook(self, l):
        def f(mod, args):
            x = args[0]; self.z[l] = x
            if self.gains is not None:                                   # gains: [B, L, H]
                B, T, _ = x.shape
                x = (x.view(B, T, self.H, self.D) * self.gains[:, l, None, :, None]).view(B, T, -1)
                return (x,)
        return f

    def forward(self, ids, att, gains=None, want_attn=False):
        self.gains = gains
        o = self.m(input_ids=ids, attention_mask=att, output_attentions=want_attn, output_hidden_states=want_attn)
        self.gains = None
        return o


def last_index(att):
    return att.sum(1) - 1                                                # right padding: last real token


def features(fz, ids, att, regions, cand_ids):
    """regions: [B, N_REGION, T] float masks; cand_ids: [B, 2] token ids of the two candidate names. Returns [B, F]."""
    with torch.no_grad():
        o = fz.forward(ids, att, want_attn=True)
        pre_lnf = fz.pre_lnf                                                               # save before lens calls ln_f again
        B = ids.shape[0]; li = last_index(att); ar = torch.arange(B)
        # 1+2. attention of every head at the answer position, summed over each region
        A = torch.stack([o.attentions[l][ar, :, li, :] for l in range(fz.L)], 1)          # [B, L, H, T]
        f_att = torch.einsum("blht,brt->blhr", A, regions).reshape(B, -1)                  # [B, L*H*R]
        # 3. logit lens: margin cand1 - cand2 after every floor
        WU = fz.m.lm_head.weight; lnf = fz.m.transformer.ln_f
        dirv = WU[cand_ids[:, 0]] - WU[cand_ids[:, 1]]                                     # [B, d]
        hs = list(o.hidden_states[:-1])                                                   # embeddings .. input of last floor
        f_lens = torch.stack([(lnf(h[ar, li]) * dirv).sum(-1) for h in hs] + [o.logits[ar, li, cand_ids[:, 0]] -
                                                                             o.logits[ar, li, cand_ids[:, 1]]], 1)  # [B, L+1]
        # 4. tug of war: each head's direct write toward cand1 - cand2 (scaled by the final layer-norm of this run)
        xp = pre_lnf[ar, li]                                                             # residual before ln_f
        scale = ((xp - xp.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True).add(lnf.eps).sqrt()
        f_dla = []
        for l in range(fz.L):
            W = fz.m.transformer.h[l].attn.c_proj.weight                                   # [in, out]
            z = fz.z[l][ar, li].view(B, fz.H, fz.D)
            contrib = torch.einsum("bhd,hdo->bho", z, W.view(fz.H, fz.D, -1))              # [B, H, d]
            contrib = contrib - contrib.mean(-1, keepdim=True)                             # ln_f centres the residual
            f_dla.append((contrib * (dirv * lnf.weight)[:, None, :]).sum(-1))
        f_dla = torch.stack(f_dla, 1).reshape(B, -1) / scale
        return torch.cat([f_att, f_lens, f_dla], 1), o.logits[ar, li]


class Observer(nn.Module):
    def __init__(self, n_feat, L, H, hidden=128):
        super().__init__()
        self.L, self.H = L, H
        self.norm = nn.LayerNorm(n_feat)
        self.body = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(0.1))
        self.alarm = nn.Linear(hidden, 1)
        self.knobs = nn.Linear(hidden, L * H)
        nn.init.zeros_(self.knobs.weight); nn.init.zeros_(self.knobs.bias)       # starts as 'do nothing': all gains 1

    def note(self, f):
        """The observer's own summary of what it saw: 'hidden' numbers that the alarm and the knobs are read from."""
        return self.body(self.norm(f))

    def forward(self, f):
        h = self.note(f)
        a = torch.sigmoid(self.alarm(h)).squeeze(-1)                               # [B]
        g = 1 + a[:, None] * torch.tanh(self.knobs(h))                             # [B, L*H] in [0, 2]
        return a, g.view(-1, self.L, self.H)
