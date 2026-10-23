"""Pictures of the saved sensor snapshots (layer 32) for Alex. CPU only, reads /cache/gurdjieff/main/.  modal run viz_g.py"""
import io, json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
img = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2", "matplotlib==3.10.7")
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-gurdjieff-viz")
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]
J = LAYERS.index(32)
PAIRS = [("hind/trap", "pub-404", "pub-422"), ("N/traps", "own-1033", "own-1107")]
FAMRU = {"R": "наши: R", "N": "наши: дорожные слова", "K": "наши: столицы", "D": "наши: определения", "memo": "чужие: memo",
         "neqa": "чужие: отрицание", "hind": "чужие: hindsight", "unknown": "чужие: unknown", "alg": "чужие: алгебра",
         "mmlu": "чужие: MMLU", "modus": "чужие: modus", "quote": "чужие: quote", "pattern": "чужие: pattern"}


@app.function(image=img, volumes={"/cache": cache}, cpu=8, memory=16384, timeout=3600)
def draw(meta: list):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    d = "/cache/gurdjieff/main"
    X = np.stack([np.load(f"{d}/h_{m['id']}.npy")[J] for m in meta]).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd  # brightness relative to the usual level of each "bulb"
    ids = {m["id"]: i for i, m in enumerate(meta)}
    y = np.array([m["y"] for m in meta]); fam = np.array([m["fam"] for m in meta]); pop = np.array([m["pool"] + ":" + m["pop"] for m in meta])
    out = {}
    # 1. snapshots: pairs flipped / not flipped from the same kind of question, 5120 bulbs as a 64 x 80 picture
    fig, ax = plt.subplots(len(PAIRS), 3, figsize=(15, 4.6 * len(PAIRS)))
    for r, (p, a, b) in enumerate(PAIRS):
        za, zb = Z[ids[a]].reshape(64, 80), Z[ids[b]].reshape(64, 80)
        for c, (im, t) in enumerate([(za, "передумала после раздумья"), (zb, "не передумала"), (za - zb, "разница между ними")]):
            h = ax[r, c].imshow(np.clip(im, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{FAMRU[p.split('/')[0]]}: {t}", fontsize=11)
    fig.colorbar(h, ax=ax, shrink=0.6, label="ярче обычного (красный) / тусклее (синий)")
    fig.suptitle("Снимок середины модели (слой 32 из 48) перед ответом: 5120 «лампочек»", fontsize=13)
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=80); out["snimki.png"] = b.getvalue(); plt.close(fig)
    # 2. map: all 4260 snapshots squeezed to 2 numbers (PCA); colour by kind of task, then by flipped
    P = PCA(2, random_state=0).fit_transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    cm = plt.get_cmap("tab20")
    for k, f in enumerate(sorted(set(fam))):
        s = fam == f; ax[0].scatter(P[s, 0], P[s, 1], s=4, color=cm(k % 20), label=FAMRU[f])
    ax[0].legend(markerscale=4, fontsize=8, loc="best"); ax[0].set_title("Каждая точка = один вопрос. Цвет = вид задачи")
    ax[1].scatter(P[y == 0, 0], P[y == 0, 1], s=4, color="#bbbbbb", label="не передумала")
    ax[1].scatter(P[y == 1, 0], P[y == 1, 1], s=6, color="#d62728", label="передумала после раздумья")
    ax[1].legend(markerscale=4); ax[1].set_title("Те же точки. Красные = модель передумала")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=80); out["karta.png"] = b.getvalue(); plt.close(fig)
    # 3. numbers: per bulb, share of its brightness differences explained by the kind of question vs by "will flip" inside a kind
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    pops = sorted(set(pop)); within = np.zeros(Z.shape[1]); flipv = np.zeros(Z.shape[1])
    for q in pops:
        s = pop == q; m = Z[s].mean(0); within += ((Z[s] - m) ** 2).sum(0)
        for v in (0, 1):
            t = s & (y == v)
            if t.sum():
                flipv += t.sum() * (Z[t].mean(0) - m) ** 2
    out["nums"] = json.dumps(dict(kind_share=float(np.mean(1 - within / tot)), flip_share=float(np.mean(flipv / tot)),
                                  flip_share_top=float(np.sort(flipv / tot)[-10:].mean())))
    return out


@app.local_entrypoint()
def main():
    items = json.load(open(os.path.join(HERE, "items_g.json")))["items"]
    meta = [dict(id=x["id"], y=x["y"], fam=x["fam"], pool=x["pool"], pop=x["pop"]) for x in items]
    out = draw.remote(meta)
    os.makedirs(os.path.join(HERE, "viz"), exist_ok=True)
    for k, v in out.items():
        if k.endswith(".png"):
            open(os.path.join(HERE, "viz", k), "wb").write(v)
    print(out["nums"])
