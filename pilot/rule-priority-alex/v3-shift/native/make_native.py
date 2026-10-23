"""'Native' vs fictional words (Alex: meaning vs impact, like a native vs a learned language).
Same generator, same seed and same list lengths as make_match.py, so every set has the SAME case structure and
history as the fictional sets; only the words differ. Native words follow road common sense:
  R1 STOP if barrier=DOWN; R2 STOP if crosswalk=ACTIVE and pedestrians=PRESENT; R3 GO if light=GREEN; else CHECK.
The update changes R2 to pedestrians=ABSENT, i.e. AGAINST common sense: D1 (light GREEN, crosswalk ACTIVE,
pedestrians PRESENT) is GO under the new rule.
Arms kept: fresh (D1, D3 only), change_0, {A,B,C}_change_60 (D1-D4) -> compare with the same arms in match/results.
python3 make_native.py <out.json>
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "match"))
import make_match as mm

# same lengths as the fictional pools (6 names, 4 value pairs, 6 label triples) -> identical RNG draws
mm.F1_NAMES = ["barrier", "gate", "boom", "barrier", "gate", "boom"]
mm.F2_NAMES = ["light", "signal", "lamp", "light", "signal", "lamp"]
mm.F3_NAMES = ["crosswalk", "crossing", "zebra", "crosswalk", "crossing", "zebra"]
mm.F4_NAMES = ["pedestrians", "people", "walkers", "pedestrians", "people", "walkers"]
mm.F1_VALS = [("UP", "DOWN"), ("RAISED", "LOWERED"), ("OPEN", "CLOSED"), ("UP", "DOWN")]
mm.F2_VALS = [("GREEN", "YELLOW"), ("GREEN", "AMBER"), ("GREEN", "YELLOW"), ("GREEN", "AMBER")]
mm.F3_VALS = [("ACTIVE", "INACTIVE"), ("ON", "OFF"), ("ACTIVE", "INACTIVE"), ("ON", "OFF")]
mm.F4_VALS = [("PRESENT", "ABSENT"), ("CROSSING", "NONE"), ("PRESENT", "ABSENT"), ("CROSSING", "NONE")]
mm.LABELS = [("STOP", "GO", "CHECK"), ("HALT", "PROCEED", "SLOW"), ("STOP", "DRIVE", "CAUTION"),
             ("BRAKE", "GO", "CHECK"), ("HALT", "GO", "SLOW"), ("STOP", "PROCEED", "CAUTION")]

KEEP = {"change_0", "A_change_60", "B_change_60", "C_change_60"}
items = [x for x in mm.build() if x["arm"] in KEEP or (x["arm"] == "fresh" and x["probes"][0]["id"] in ("D1", "D3"))]
if __name__ == "__main__":
    json.dump(items, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(len(items), "prompts")
