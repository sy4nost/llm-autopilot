# Ask-twice: dev check result (2026-10-18, seed 12345, ~$0.3). Frozen rule says STOP: only 1 family with a habit.

dev_choice.txt / dev_choice.json. Right answers of the normal model on counterfactual traps:
- U units (week has 5 days etc.), all 4 variants: 96-100% right -> no habit.
- K capitals: v1 "first letter of the capital" 88% (10 habit errors of 83); v2/v3 "flew to the French capital" 100%,
  BUT the option-mass filter kept only 16/120 and 41/120 items (the model's next word after "The answer is" is mostly not a
  bare city name; e.g. Austria item mass 0.015). So K v2/v3 is a format artifact, not a clean "no habit".
- D defined words: v1 "a 'hand' has 7 fingers; how many fingers does a hand have?" 68% right, all errors = habit
  (cube 0/11, hexagon 2/9, square 5/9); v2-v4 (counting several, story, filler) 100% right.
Lesson: 14B follows a stated counterfactual almost always when it must compute with it; the habit shows up when the question
asks the familiar fact back directly (same as redefine-math "first digit of π"). My written guess (harder = more habit) was wrong.
