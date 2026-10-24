"""Post hoc (not in PREREG): bootstrap CI for the secondary PROBE vs CONF. Reuses score_cc.py as is."""
import sys, io, contextlib, os
sys.argv = ["score_cc.py", "main"]; buf = io.StringIO(); g = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_cc.py"), "__name__": "x"}
with contextlib.redirect_stdout(buf):
    exec(open(g["__file__"]).read(), g)
auc, boot, conf = g["auc"], g["boot"], g["conf"]
for strat in ("cell", "ck"):
    lo, hi = boot(conf, lambda T: auc(T, "probe", strat) - auc(T, "conf", strat))
    print(f"PROBE - CONF within {strat}: {auc(conf, 'probe', strat) - auc(conf, 'conf', strat):+.3f} [{lo:+.3f},{hi:+.3f}]")
