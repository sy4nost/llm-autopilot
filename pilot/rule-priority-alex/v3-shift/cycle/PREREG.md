# PREREG: life-cycle frame test (2026-10-19). Plan: PLAN.md v2 (review in review/plan-review.md). Written before the main run.

## Frozen (sha256)
- PLAN.md 63047a254ba026aa9a86445409ea58141f5173ddade79948f82db4cb43e96593
- make_cy.py ba3110eb4c4ce6b69358130e209283a7efc02a3bcda7e7afe739f53e5f2cfccf
- modal_cy.py 403c0d791d88da403cc714cfc4625a69f716910d5874c939df30a5c699199d20
- score_cy.py 5f8ef68698a361f64a5d7b12f78bc86cf6b50ed43681fd131c64710019548100
- sets_cy.json f45bc8a2815d732c05fc67f27e1a87d64864d87fc63e3de947f8068af992a0c6

## Run
modal run --detach modal_cy.py --sets sets_cy.json --tag main ; modal run modal_cy.py --tag main --get ; python3 -I score_cy.py out_main.jsonl > report_main.txt (reported as is).

Smoke items (78) are a subset of main items and were seen in smoke output (smoke v1 and v2). No settings were tuned.
