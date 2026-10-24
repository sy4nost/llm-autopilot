"""Download helper (added after the main run; does not touch scores): compacts /cache/cicmlamp/<tag>/out.json on Modal once
(drops layer-24/40 residuals, which score_cc.py does not use; layer-32 residual kept as float16), then fetches the pre-gzipped bytes in chunks
and rebuilds out_<tag>.json locally in the format score_cc.py reads (hid = [[], layer32, []]).
  modal run fetch_cc.py --tag main"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-cicmfetch", image=modal.Image.debian_slim(python_version="3.12").pip_install("numpy"))


@app.function(volumes={"/cache": cache}, timeout=1800, memory=8192)
def compact(tag: str):
    import gzip, numpy as np
    d = f"/cache/cicmlamp/{tag}"; out = json.load(open(f"{d}/out.json"))
    h = np.array([o["hid"][1] for o in out], dtype=np.float16)
    for o in out:
        o.pop("hid")
    blob = gzip.compress(json.dumps(dict(rows=out, done=os.path.exists(f"{d}/DONE"))).encode() + b"\n@@HID@@\n" + h.tobytes())
    open(f"{d}/compact.gz", "wb").write(blob); cache.commit()
    return len(blob)


@app.function(volumes={"/cache": cache}, timeout=600)
def chunk(tag: str, part: int):
    import base64
    b = open(f"/cache/cicmlamp/{tag}/compact.gz", "rb").read(); C = 1_000_000
    return json.dumps(dict(c=base64.b64encode(b[part * C:(part + 1) * C]).decode(), more=(part + 1) * C < len(b)))


@app.local_entrypoint()
def main(tag: str = "main"):
    import base64, gzip, numpy as np
    print("compact bytes", compact.remote(tag))
    parts, k = [], 0
    while True:
        r = json.loads(chunk.remote(tag, k)); parts.append(base64.b64decode(r["c"])); k += 1
        if not r["more"]:
            break
    raw = gzip.decompress(b"".join(parts)); js, hb = raw.split(b"\n@@HID@@\n", 1)
    meta = json.loads(js); h = np.frombuffer(hb, dtype=np.float16).reshape(len(meta["rows"]), -1)
    for o, v in zip(meta["rows"], h):
        o["hid"] = [[], [round(float(x), 3) for x in v], []]
    json.dump(meta["rows"], open(os.path.join(HERE, f"out_{tag}.json"), "w"))
    if meta["done"]:
        open(os.path.join(HERE, f"out_{tag}.DONE"), "w").write("1")
    print("rows", len(meta["rows"]), "done", meta["done"])
