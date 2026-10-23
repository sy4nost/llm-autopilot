# "Native" words vs made-up words (14B, 2026-10-16)

Alex's idea: the meaning of a word and its effect are not the same thing. A swear word in your native language hits
harder than in a learned one, because the native language is built in as experience. The test: a rule on words for
which the model has "native" experience from training, vs the same rule on made-up words.

## Setup

- Same structure, same generator and same seed as the A/B/C test (`../match/make_match.py`): each set has the same
  cases and the same history; only the words change. This is a paired comparison.
- "Native" words, road common sense: STOP if the barrier is down (barrier=DOWN); STOP if the crosswalk is active and
  there are pedestrians on it (crosswalk=ACTIVE and pedestrians=PRESENT); GO if the light is green; otherwise CHECK.
  Names and labels vary slightly (gate/boom, lamp/signal, HALT/PROCEED etc.).
- The rule change goes **against common sense**: the new Rule 2 says STOP if there are **no** pedestrians. Then D1
  (green, crosswalk active, pedestrians present) under the new rule = GO.
- Conditions: gate (the new rule directly in the text, no change), change without history, A/B/C with history 60.
  540 prompts × 5 answers, H100. Files: `make_native.py`, `sets_native.json`, `results/`, `score_native.py`,
  `score_native_out.txt`. Made-up words are taken from `../match/results/`.

## Result: old-rule answers on D1+D3

| | made-up words | "native" words | difference (95% interval) |
|---|---|---|---|
| gate (new rule directly) | 0% | 11% | +11% [+3, +21] |
| change, no history | 37% | **81%** | **+44% [+28, +59]** |
| change, 60 examples, exact copy (A) | 93% | 100% | +7% [+2, +13] |
| change, 60 examples, no copy (B) | 38% | 71% | +34% [+14, +53] |
| change, 60 examples, no relevant examples (C) | 12% | 41% | +29% [+16, +42] |

Where the new rule adds STOP (D2, D4), "native" words help instead: 85–100% correct.

## What it means

1. **Built-in experience works like history even when there is no history.** Without a single example, only with an
   announced change, 14B answers the old way on "native" words in 81% of cases (37% on made-up words). So knowledge
   from training pulls towards the old answer as strongly as 60 examples with an exact copy pull on made-up words.
2. **The two sources of habit add up.** Without an exact copy (B) and with no examples of the old rule at all (C),
   "native" words give +29…+34% old answers. With an exact copy (A) the ceiling is almost reached on both sides (93% and 100%).
3. **The model can apply an "unnatural" rule if it is simply given.** In the gate (the new rule directly, without
   "it was like this, now it is like that") there are only 11% old answers. What breaks is the transition: when the
   familiar rule is given first and then it is said that it changed.

Answer to the question "what kind of experience do we need": the model has two kinds of experience, and both pull
towards the past answer:
- a precedent in the text (an exact copy of the case), almost 100%;
- built-in experience from training ("native language"), which even without examples outweighs an explicitly
  announced change in 4 of 5 cases.
So a "lamp" test must work on both, and tasks on "native" words are the hardest for it and the closest to real life.

## Caveats

- The gate on "native" words is below 85%: D1 76%, D3 54% correct (only 11% old answers, the rest a third label).
  The unnatural rule itself is hard for the model, but it almost never gives the habitual answer in the gate.
- One topic (road), one model.
