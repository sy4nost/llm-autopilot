# Check of the "conflict" guess (written 2026-10-19 before running conflict_g.py)

Guess: the own-trained sensor senses how strongly the stated fiction contradicts the model's knowledge.

Conflict axis (built WITHOUT flip labels): in each own family, mean layer-32 snapshot of trap items (fiction contradicts the fact)
minus mean snapshot of agree items (fiction agrees with the fact); axis = average of these family differences (unit length).
Leave-one-family-out: the axis used on family f is built from the other three families only.

Measured on own trap populations with >= 10 flips and >= 10 non-flips (R traps, R cfright, R compute, D traps, K traps; N traps has 10 non-flips):
- A: within-population AUC of the projection on the held-out conflict axis for y (THINK != DIRECT).
- B: same for the LOFO sensor (as before), and for the LOFO sensor retrained on snapshots with the conflict axis removed.
- C: cosine between the sensor's direction and the conflict axis.
- D (public): projection on the all-family axis, within-population AUC on public trap populations; mean projection traps vs controls.

"Guess supported" if: mean A >= 0.65 and >= low-confidence on those populations, AND removing the axis lowers the mean LOFO sensor AUC
by >= 0.05. "Guess not supported" if mean A < 0.60 or removal changes the sensor by < 0.02. Otherwise "partly".
Sign of the projection is fixed (trap direction = higher), never chosen per population.
amended 2026-10-19: progress prints only (first run cut off by client timeout, no output)
