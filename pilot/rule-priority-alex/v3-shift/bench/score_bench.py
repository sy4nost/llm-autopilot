"""Score MMLU / GSM8K normal vs heads off.  python3 -I score_bench.py bench_<model>.jsonl
Accuracy per setting, paired difference with a bootstrap over questions, and how many answers flipped each way."""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
rng = random.Random(1)
mean = lambda v: sum(v) / len(v)
for bench in ("mmlu", "gsm8k"):
    rs = [r for r in rows if r["bench"] == bench]
    if not rs:
        continue
    print(f"\n{bench}: {len(rs)} questions")
    for s in rs[0]["pred"]:
        print(f"  {s:8} {mean([r['pred'][s] == r['gold'] for r in rs]):.1%}")
    for s in [k for k in rs[0]["pred"] if k != "normal"]:
        d = [(r["pred"][s] == r["gold"]) - (r["pred"]["normal"] == r["gold"]) for r in rs]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"  {s} - normal: {mean(d):+.1%} [{b[100]:+.1%}, {b[3899]:+.1%}]; "
              f"fixed {sum(x > 0 for x in d)}, broken {sum(x < 0 for x in d)}")
