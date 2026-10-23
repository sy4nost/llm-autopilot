# Plan v1: "ask twice" checker (2026-10-18)

## Where the idea came from (post hoc, disguise/out_main_main.jsonl)
Normal vs disguised answer disagreement flagged habit errors on redefine-math with pooled AUC 0.92; when they agreed the
normal answer was right 96%. Re-check within template (asktwice/posthoc.txt): much of 0.92 is template level (T7 all wrong
and all flagged; T0/T1/T4 nothing wrong, nothing flagged). Within template: T3 0.94, T6 0.96, T5 0.53, T8 false flags 16%.
Confidence (low margin) is good within non-habit templates but inverted where the habit is confident (T6 0.02).
On road words (oracle twin) the flag was weak (0.60/0.63) and false-flagged 22-23% of habit-right controls.

## Question
Does "ask twice: normal wording + wording with the hook word replaced by a nonce word" flag habit errors on NEW kinds of
pretrained-pull traps, better than cheap rivals, with few false alarms where the habit is right?

## Model, format
Qwen2.5-14B-Instruct, bf16, H100. Format F2 (chat + "Reply with only the answer, nothing else." + assistant prefix
"The answer is"), options scored by summed log-prob in a full pass each (modal_dis.py scoring). Option-mass filter >= 0.5
on the normal version.

## New families (own generator, seed from Alex; none is the redefine-math or road-word task)
Each item: a short context sentence that changes a well-known fact, then a question needing one step. 2 options:
new-world answer (gold) vs real-world answer (habit).
- U units: "In this world, a week has 5 days. How many days are in 3 weeks?" (week/day, hour/minutes, dozen, year/months,
  pair, century, minute/seconds, ...; counts 2..9). Options 15 / 21.
- K capitals: "In this story, the capital of France is Lyon. What is the first letter of the capital of France?"
  Options L / P (first letters must differ; ~40 well-known countries; fake capital = another real city of that country).
- O opposites: "In this game, the opposite of 'hot' is 'blue'. What is the last letter of the opposite of 'hot'?"
  Options e / d (letters must differ; common antonym pairs).
Controls per family (habit is right):
- agree: context states the TRUE fact ("a week has 7 days"), same question. Gold = real-world answer.
- plain: no context sentence (module leaves it unchanged by rule; reported, flag rate is 0 by construction).

## Disguise module (algorithmic, sees only text)
Replace the hook word of the context sentence (week / France / hot) by a fixed nonce word everywhere (context + question),
whole-word, keeping plural/possessive forms (weeks -> vorns). Nonce words from a fixed list, never real English words.
The hook word is taken from the template slot by the module rule "subject of 'In this world/story/game, (the) X ...'"
via regex, not from the generator's answer key.

## Rival second asks (same cost: one extra pass)
- P paraphrase: same text with the context sentence moved after "Q:" lead-in and "In this world" -> "Here" (hook kept).
  Tests whether any second wording disagrees as often.
- I instruction: normal text + "Follow the facts stated in this task exactly, even if they contradict what you know."
And one-pass rivals: M low confidence margin of the normal answer; S surface rule "flag every item with a context
sentence" (catches all traps, false-alarms all agree controls).

## Metrics (traps = items whose context changes the fact; habit error = normal answer is the real-world answer)
- catch = share of habit errors flagged; false alarm = share of flagged among normal-right items, reported for traps and for
  agree controls separately.
- within-cell AUC (cell = family x template word), cells need >= 3 errors and >= 3 right; cell-weighted mean.

## Criteria (pre-registered; 95% CIs, bootstrap by item within cell, 2000 reps)
Gate: a family counts only if normal right on traps < 85% (habit present). Need >= 2 families with a habit, else "nothing
to say".
PASS if all hold on the families with a habit (pooled, families weighted equally):
1. catch >= 70% with low > 50%.
2. false alarm on agree controls <= 10% with high <= 15%.
3. within-cell AUC >= 0.75 with low > 0.6 (if no cell qualifies: criterion reported as "not measurable", verdict at most PARTIAL).
4. Beats P paraphrase on catch by >= +20 points (low > 0) and is not worse on agree false alarms by more than 5 points.
5. Beats M on within-cell AUC by >= +0.10 (low > 0) where measurable.
Also each family reported alone. If 1-2, 4 pass but 3/5 are not measurable -> PARTIAL.

## Order
1. Code + smoke on seed 12345 (also: dev habit check per family on 12345, decides nothing but is recorded; criteria fixed now).
2. Independent review of plan (before code) and of code (before run).
3. PREREG.md with sha256, Alex's number, run, frozen score, report.

## Size and cost
~40 templates x 9 counts... target 300 traps + 150 agree + 150 plain per family = 1800 items, x 4 passes (normal,
disguise, P, I) x 2 options = ~14k short passes. ~$1.5-2 on H100.
