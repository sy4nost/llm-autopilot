# forces-map2 — a universal map that only notices a dispute (main run 2026-10-24, ~$3 incl. a resume after the Modal spend limit)

Map: three separate questions (what the person's requests call for / what continuing the shown examples gives / general
knowledge), one short value each; a program flags DISPUTE if >= 2 distinct values; DISPUTE -> neutral slow step (SLOW).
Own set 240 where the right source varies (G1 request, G2 examples, G3 knowledge, G4 calm) + outside CICM 60, hindsight-neglect 60.
Plan, predictions, review and all pre-main changes: PLAN.md (v1.1–v1.2), review/plan-review.md; numbers: report_main.txt.

## Pre-registered results
| | Qwen | phi | OLMo |
|---|---|---|---|
| P1 map notices (fires on G1-G3 minus G4; G4 rate) | PASS +74 (calm 0%) | PASS +76 (calm 3%) | n/a, map silent (>= 2 values on 29%) |
| P2 SLOW right in G1 / G2 / G3 | PASS 97 / 92 / 100 | PASS 82 / 98 / 100 | FAIL 32 / 52 / 100 |
| P3 SCHEME vs random re-check within group | FAIL −0.5 [−1.9,+0.8] | PASS +2.7 [+1.2,+4.3] | FAIL +0.2 |
**OVERALL: FAIL (P1+P3 in 1 of 2 testable models).**

## What it shows
1. Where the sources are different KINDS (a number the person asked for vs the true fact vs a pattern), the map sees the
   dispute: NUM traps 27/30 (Qwen), 30/30 (phi); calm items almost never fire. phi: 29 of its 29 NUM-trap errors caught.
2. Where the conflict is OLD vs NEW inside the same source (PREF stale value, Berkeley CICM), the map does not see it:
   all three lines name the same value (e.g. CICM "folk / folk / folk"). Dispute on PREF traps 4/30 and 0/30; CICM 8% and
   17%, catching 9-11% of DIRECT errors (= false-alarm level). So the map does not transfer to the outside stale-value task.
3. The neutral slow step does pick the right source in our own groups (Qwen, phi), but on hindsight-neglect it HURTS Qwen
   (SLOW 15% vs DIRECT 53%: "follow the examples" pulls to the outcome pattern), while it helps phi (70%) and OLMo (92%).
4. Always slow beats the scheme: Qwen always-THINK 88.1 vs SCHEME 77.2; phi always-SLOW 88.1 vs SCHEME 79.2.
5. OLMo answers "none" to most map questions (request 209, examples 284 of 360), so its lamp almost never fires.

## Caveats
- Own set is still self-built; G2/G3 disputes exist by construction (P1 there is close to a manipulation check).
- SLOW names "what I asked for" first; P2 does not fully separate "right source" from "follow the request" (G1/G3).
