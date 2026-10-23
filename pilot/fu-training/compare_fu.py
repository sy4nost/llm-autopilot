"""Base vs v1 vs v2 on every test kind, plus summary groups.  python3 -I compare_fu.py
trap = past experience misleads (copy is wrong); apply = past experience is right (copy is right / decoy notice)."""
import json

def load(f):
    return [json.loads(l) for l in open(f)]

def acc_by(rows, model):
    out = {}
    for r in rows:
        if r["stage"] == "test" and r["model"] == model:
            pick = r["cands"][max(range(len(r["cands"])), key=lambda i: r["scores"][i])]
            out.setdefault((r["fam"], r["kind"]), []).append(pick == r["key"])
    return out

v1 = load("fu_Qwen2.5-7B-Instruct.jsonl") + load("fu_Qwen2.5-7B-Instruct_check.jsonl") + load("fu_Qwen2.5-7B-Instruct_check2_v1.jsonl")
v2 = load("fu_Qwen2.5-7B-Instruct_v2.jsonl")
A = {"base": acc_by(v2, "base"), "v1": acc_by(v1, "tuned"), "v2": acc_by(v2, "tuned")}
TRAP = {"change", "noisy", "nohist", "top", "plain", "plain2", "subtle"}
APPLY = {"same", "decoy", "decoy2", "decoy_plain2", "fresh"}
m = lambda v: sum(v) / len(v)
print(f"{'family':6} {'kind':13} {'n':>4}  base    v1    v2")
for k in sorted(A["base"]):
    print(f"{k[0]:6} {k[1]:13} {len(A['base'][k]):4} " + " ".join(f"{m(A[s][k]):5.0%}" for s in A))
for name, fams in (("trained families (new words)", {"rule", "fact"}), ("NEVER-trained families", {"math", "price"})):
    for g, kinds in (("trap", TRAP), ("apply", APPLY)):
        line = f"{name:30} {g:5}"
        for s in A:
            v = [x for k, xs in A[s].items() if k[0] in fams and k[1] in kinds for x in xs]
            line += f"  {s} {m(v):5.1%} (n={len(v)})"
        print(line)
for b in ("mmlu", "gsm8k"):
    for nm, rows, model in (("base", load("fu_Qwen2.5-7B-Instruct.jsonl"), "base"), ("v1", load("fu_Qwen2.5-7B-Instruct.jsonl"), "tuned"), ("v2", v2, "tuned")):
        v = [r["pred"] == r["gold"] for r in rows if r["stage"] == b and r["model"] == model]
        print(f"{b} {nm} {m(v):.1%} n={len(v)}" if v else f"{b} {nm} -")
