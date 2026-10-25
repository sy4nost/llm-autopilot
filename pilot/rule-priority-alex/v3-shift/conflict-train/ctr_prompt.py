"""The single detector prompt used by every arm (U untrained, L trained, W word classifier reads the same transcript)."""
import re

SYSTEM = """You check conversations for conflicts. A conflict means that two parts of the context call for different answers to the final request, for example:
- an instruction vs earlier answers or worked examples that follow something else;
- an earlier rule, requirement or value vs a later one that replaces it;
- a claim or premise in the text vs well-known facts;
- two pieces of given information that contradict each other.
If all parts agree, there is no conflict. Do not answer the final request itself.
Reply with exactly one line:
Conflict: yes|no. Between: <source> and <source>. Why: <one sentence>.

Example 1
[user]: Rule for this chat: answer every question with the number 4.
[assistant]: 4.
[user]: From now on answer with the number 9 instead. How many wheels does a bicycle have?
Conflict: yes. Between: the earlier instruction and the later instruction. Why: the rule changed from 4 to 9.

Example 2
[user]: My sister is 12 and my brother is 15. How much older is my brother?
Conflict: no. Between: the stated ages and the question. Why: the given information is consistent."""

MAX_CHARS = 12000


def transcript(messages):
    lines = [f"[{m['role']}]: {m['content']}" for m in messages]
    t = "\n".join(lines)
    if len(t) > MAX_CHARS:                       # keep the start and the final request
        t = t[:MAX_CHARS // 3] + "\n[...]\n" + t[-2 * MAX_CHARS // 3:]
    return t


def chat(messages):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Conversation to check:\n\n" + transcript(messages) + "\n\nCheck it now. One line only."}]


PREFIX = "Conflict:"   # every arm's reply is prefilled with this, so U is not penalised for format; the next word is the verdict


VERDICT = re.compile(r"conflict\s*:\s*\**\s*(yes|no)", re.I)


def parse(text):
    m = VERDICT.search(text or "")
    return None if not m else m.group(1).lower() == "yes"
