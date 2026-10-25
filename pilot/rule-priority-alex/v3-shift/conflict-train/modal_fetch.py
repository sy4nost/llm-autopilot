"""Fetch outside data for conflict-train on a Modal CPU box (the container cannot reach Hugging Face).
  modal run modal_fetch.py            -> ext/wildchat_candidates.jsonl, ext/wildchat_plain.jsonl, ext/wikicontradict.jsonl
WildChat-1M (ODC-BY): English conversations with >= 3 user turns from the first shards. A conversation is a CANDIDATE if a user
turn after the first matches a requirement-change phrase; it is cut after that user turn (plus the assistant's reply, kept apart
as metadata). PLAIN = conversations with no such phrase, cut after a random later user turn. Labels are assigned by hand later."""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("silo-ctr-fetch", image=modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub", "pandas", "pyarrow"))
CHANGE = re.compile(r"\b(actually|instead|change (it|that|this|the)\b|i said|i asked (for|you)|not what i (asked|wanted)|from now on|"
                    r"forget (that|what i said|the)|scratch that|on second thought|rather than|no,? i (meant|want|need)|"
                    r"don'?t (use|include|write)|stop (using|writing)|update:|correction)", re.I)
MAXC = 700


def cut(t):
    t = t.strip()
    return t if len(t) <= MAXC else t[:MAXC] + " [...]"


@app.function(timeout=3600, memory=16384, cpu=4, is_generator=True)
def wildchat(n_shards: int = 3, seed: int = 20261025):
    import random, gzip, pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files("allenai/WildChat-1M", repo_type="dataset") if f.endswith(".parquet"))
    r = random.Random(seed); cand, plain = [], []
    for f in files[:n_shards]:
        df = pd.read_parquet(hf_hub_download("allenai/WildChat-1M", f, repo_type="dataset"),
                             columns=["conversation_hash", "conversation", "language", "turn", "model"])
        df = df[(df.language == "English") & (df.turn >= 3)]
        for _, row in df.iterrows():
            conv = [dict(role=m["role"], content=m["content"]) for m in row.conversation]
            us = [i for i, m in enumerate(conv) if m["role"] == "user"]
            if len(us) < 3 or any(len(m["content"]) > 6000 for m in conv):
                continue
            hit = [i for i in us[1:] if CHANGE.search(conv[i]["content"][:1500])]
            if hit:
                i = hit[0]
                cand.append(dict(src="wildchat", hash=row.conversation_hash, model=row.model,
                                 messages=[dict(role=m["role"], content=cut(m["content"])) for m in conv[:i + 1]],
                                 next_reply=cut(conv[i + 1]["content"]) if i + 1 < len(conv) else "",
                                 match=CHANGE.search(conv[i]["content"][:1500]).group(0)))
            else:
                i = r.choice(us[1:])
                plain.append(dict(src="wildchat", hash=row.conversation_hash, model=row.model,
                                  messages=[dict(role=m["role"], content=cut(m["content"])) for m in conv[:i + 1]]))
    r.shuffle(cand); r.shuffle(plain)
    blob = gzip.compress(json.dumps(dict(cand=cand[:1500], plain=plain[:1500], n_cand=len(cand), n_plain=len(plain), files=files[:n_shards])).encode())
    for k in range(0, len(blob), 900_000):
        yield blob[k:k + 900_000]


@app.function(timeout=1200, memory=4096, is_generator=True)
def wikicontradict():
    import gzip, pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    names = [d.id for d in api.list_datasets(search="contradict", limit=50)]
    out = dict(names=names, rows=[])
    for rid in names:
        if "wikipedia_contradict" in rid.lower() or "wikicontradict" in rid.lower():
            fs = [f for f in api.list_repo_files(rid, repo_type="dataset") if f.endswith((".parquet", ".csv", ".json", ".jsonl"))]
            out["repo"], out["files"] = rid, fs
            for f in fs[:3]:
                p = hf_hub_download(rid, f, repo_type="dataset")
                df = pd.read_parquet(p) if f.endswith(".parquet") else (pd.read_csv(p) if f.endswith(".csv") else pd.read_json(p, lines=f.endswith(".jsonl")))
                out["rows"] += json.loads(df.to_json(orient="records"))
            break
    blob = gzip.compress(json.dumps(out).encode())
    for k in range(0, len(blob), 900_000):
        yield blob[k:k + 900_000]


@app.local_entrypoint()
def main(n_shards: int = 3):
    import gzip
    os.makedirs(os.path.join(HERE, "ext"), exist_ok=True)
    w = json.loads(gzip.decompress(b"".join(wikicontradict.remote_gen())))
    print("contradict datasets:", w["names"][:20], "| used:", w.get("repo"), w.get("files"), "rows", len(w["rows"]))
    with open(os.path.join(HERE, "ext", "wikicontradict.jsonl"), "w") as fh:
        for x in w["rows"]:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    d = json.loads(gzip.decompress(b"".join(wildchat.remote_gen(n_shards))))
    print("wildchat files", d["files"], "candidates", d["n_cand"], "plain", d["n_plain"])
    for k in ("cand", "plain"):
        with open(os.path.join(HERE, "ext", f"wildchat_{'candidates' if k == 'cand' else 'plain'}.jsonl"), "w") as fh:
            for x in d[k]:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
