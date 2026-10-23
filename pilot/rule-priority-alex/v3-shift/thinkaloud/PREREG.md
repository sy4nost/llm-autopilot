# PREREG: think-aloud test (2026-10-18). Plan: PLAN.md v2 + v2.1 (reviews in review/plan-review.md).
Written before the technical smoke. Any change after the smoke is listed below with new sums.

## Frozen code (sha256)
- make_ta.py   00f21808d86b01dc51828060b2ad5dc95f89e659a5de41c775a8463e589ebd5e
- modal_ta.py  b37344db147bade4354cbfc986489a9f150bb4613015454f5f1a1bcb74bf91bd
- score_ta.py  3b58c4cc24839cfa46a4f3670bc33ce3910a85b5afc46c5805b681d77d72dcd3

## Run
1. Smoke: python3 -I make_ta.py 12345 --small -> sets_smoke.json; modal run modal_ta.py --sets sets_smoke.json --tag smoke;
   python3 -I score_ta.py out_smoke.jsonl (pipeline only; outputs kept).
2. Alex's number S: python3 -I make_ta.py S -> sets_ta.json (sha256 recorded here before the run).
3. modal run modal_ta.py --sets sets_ta.json --tag main ; python3 -I score_ta.py out_main.jsonl -> report_main.txt, reported as is.

## Smoke (seed 12345, 146 items, out_smoke.jsonl, report_smoke.txt; pipeline check only)
Pipeline runs end to end; 1 unparsed of 146; no truncation; parsed = forced answer 145/145.
Batched vs single greedy on 20 prompts: identical text 8/20, same final answer line 20/20 (bf16 batch numerics; accepted).
## Changes after the smoke
none (code sums above unchanged).

## Alex's number (2026-10-18): 2
sets_ta.json sha256 5b0654f3a278da94557825145e62a7934c9896d827be65899957f2242650c9d8 (N generator seeds: make_dis.seeds(2, 2) = 2, 3).
