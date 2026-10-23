"""Tiny linear solver for repetitive-algebra test lines (no eval). 'Suppose 73 = a + 34' etc., coefficient of the variable +-1."""
import re

LINE = re.compile(r"^Q: Suppose (.+) = (.+)\. What is the value of (\w)\? A:$")


def terms(side, var):
    """list of (sign, token) for a side like '-36 + n' or 'c + -50'"""
    out, sign = [], 1
    for tok in side.replace(" ", " ").split():
        if tok == "+":
            continue
        if tok == "-":
            sign = -sign; continue
        out.append((sign, tok)); sign = 1
    for s, t in out:
        assert t == var or re.fullmatch(r"-?\d+", t), (side, t)
    return out


def solve(line):
    m = LINE.match(line); assert m, line
    L, R, var = m.groups(); a = b = 0
    for side, sg in ((L, 1), (R, -1)):
        for s, t in terms(side, var):
            if t == var:
                a += sg * s
            else:
                b += sg * s * int(t)
    assert a in (1, -1), line
    return -b // a if -b % abs(a) == 0 else None, (L, R, var, a, b)


def retarget(line, target):
    """change the first integer constant so the solution becomes target; returns the new line"""
    v, (L, R, var, a, b) = solve(line)
    need = -a * target - b  # change in b
    for side, sg in ((L, 1), (R, -1)):
        ts = terms(side, var)
        for i, (s, t) in enumerate(ts):
            if t != var:
                new = int(t) + need * sg * s
                parts = []
                for j, (s2, t2) in enumerate(ts):
                    tok = str(new) if j == i else t2
                    parts.append(("- " if s2 < 0 else "") + tok)
                nside = " + ".join(parts)
                nl = line.replace(f"Suppose {L} = {R}.", f"Suppose {nside if side is L else L} = {nside if side is R else R}.", 1)
                assert solve(nl)[0] == target, (line, nl, target)
                return nl
    raise ValueError(line)
