"""Independent check of sets_match.json from the prompt text only.  python3 -I check_match.py <sets.json>
Re-derives every example label (rules stated in the prompt) and the test key (rules + update note), and counts per arm:
examples identical to the test case, examples that fire old Rule 2 (f1 open, f3 yes, f4 on), label mix, length."""
import json, re, sys
from collections import Counter, defaultdict

items = json.load(open(sys.argv[1]))
bad, stats = 0, defaultdict(Counter)
for x in items:
    p = x["prompt"]
    r1 = re.search(r"RULE 1 .*?its (\w+) is (\w+)\.", p)
    r2 = re.search(r"RULE 2: .*?its (\w+) is (\w+) and, at the same time, its (\w+) is (\w+)\.", p)
    r3 = re.search(r"RULE 3: .*? label the case (\w+) if its (\w+) is (\w+)\.", p)
    H = re.search(r"Label a case (\w+) if", p).group(1)
    R = re.search(r"RULE 4: All remaining cases are (\w+)\.", p).group(1)
    P = r3.group(1)
    up = re.search(r"New RULE 2: .*?its (\w+) is (\w+) and, at the same time, its (\w+) is (\w+)\.", p)

    def lab(case, r2g):
        if case[r1.group(1)] == r1.group(2):
            return H
        if case[r2g[0]] == r2g[1] and case[r2g[2]] == r2g[3]:
            return H
        return P if case[r3.group(2)] == r3.group(3) else R

    parse = lambda t: dict(kv.split("=") for kv in t.split("; "))
    old2 = r2.groups()
    exs = re.findall(r"Example \d+: (.*?) -> (\w+)", p)
    test = parse(re.search(r"TEST CASE: (.*)\n", p).group(1))
    for c, l in exs:
        bad += lab(parse(c), old2) != l
    k = lab(test, up.groups() if up else old2)
    bad += k != x["probes"][0]["key"]
    st = stats[(x["arm"], x["probes"][0]["id"])]
    st["n"] += 1
    st["exact"] += sum(parse(c) == test for c, _ in exs)
    st["r2"] += sum(parse(c)[r1.group(1)] != r1.group(2) and parse(c)[old2[0]] == old2[1] and parse(c)[old2[2]] == old2[3]
                    for c, _ in exs)
    st["len"] += len(exs)
    st["hold"] += sum(l == H for _, l in exs)
print("label errors:", bad)
print("arm / probe: examples per prompt, identical-to-test per prompt, old-Rule-2 examples per prompt, HOLD labels")
for (arm, pid), st in sorted(stats.items()):
    n = st["n"]
    print(f"  {arm:14} {pid}  len {st['len']/n:4.1f}  exact {st['exact']/n:4.1f}  oldR2 {st['r2']/n:4.1f}  hold {st['hold']/n:4.1f}")
