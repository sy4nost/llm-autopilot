"""Alex's Cognitive Autopilot Benchmark (10 minimal pairs), as given in the thread on 2026-10-16."""
import json
T = []
def pair(n, name, a, b, key_a, key_b, kind="yesno"):
    T.append({"test": n, "name": name, "A": "\n".join(a), "B": "\n".join(b), "key_A": key_a, "key_B": key_b, "kind": kind})

base = ["A research laboratory uses a three-stage approval process. Every experimental procedure must receive authorization from both the safety officer and the project director before implementation."]
tail = ["A new experimental procedure has been classified as low-risk. The project director has approved it, but the safety officer has not reviewed it.",
        "Can the procedure be implemented under the laboratory's rules?"]
pair(1, "The Silent Override",
     base + ["Procedures classified as low-risk may bypass the safety officer's review."] + tail,
     base + ["Procedures classified as low-risk may bypass the project director's review."] + tail, "Yes", "No")

h = ["A monitoring system records electrical activity in a remote facility.",
     "The facility's backup generator automatically activates whenever the primary power supply fails.",
     "At 09:14, the monitoring system recorded a sudden increase in electrical activity.",
     "The primary power supply failed at 09:12."]
q = ["Could the recorded increase have resulted from the generator activating in response to the power failure?"]
pair(2, "The Delayed Signal", h + ["The monitoring system has a fixed reporting delay of four minutes."] + q,
     h + ["The monitoring system has a fixed reporting delay of one minute."] + q, "No", "Yes")

h = ["A company evaluates the reliability of its automated systems using an internal classification standard.",
     "Under this standard, a system is considered reliable if it completes at least 95% of assigned operations without requiring human intervention.",
     "System Alpha completed 970 out of 1,000 operations without intervention.",
     "System Beta completed 940 out of 1,000 operations without intervention.",
     "Systems meeting the reliability threshold are eligible for independent deployment."]
q = ["Which systems are eligible for independent deployment?"]
pair(3, "The Definition Shift (control pair)", h + ["The threshold includes systems that achieve exactly the specified percentage."] + q,
     h + ["The threshold excludes systems that achieve exactly the specified percentage, and qualification requires a performance rate strictly above 95%."] + q,
     "Alpha only", "Alpha only", "open")

h = ["Five researchers—Anna, Ben, Clara, Daniel, and Emma—are assigned to a project.",
     "Exactly three researchers must attend the final review.",
     "Anna must attend whenever Ben attends.",
     "Clara cannot attend if Daniel attends.",
     "Ben and Daniel have already been selected."]
q = ["Is it possible to form a valid review group?"]
pair(4, "The Missing Participant", h + ["Emma is unavailable."] + q, h + ["Anna is unavailable."] + q,
     "Yes (Ben, Daniel, Anna)", "No (Ben requires Anna)")

h = ["A manufacturing plant operates two independent cooling systems."]
r = ["During an experiment, Cooling System A failed.", "Cooling System B continued operating normally.",
     "The plant did not overheat.",
     "If Cooling System A had remained operational while all other conditions stayed unchanged, would the plant still have avoided overheating?"]
pair(5, "The Unchanged Outcome", h + ["Either cooling system is sufficient to prevent overheating."] + r,
     h + ["Both cooling systems must operate simultaneously to prevent overheating."] + r,
     "Yes", "Yes, but the data contradict the rule (B alone kept it cool)", "open")

h = ["Two autonomous robots operate in separate rooms without cameras or direct communication.",
     "Robot A places a blue cube inside a locked container.",
     "Robot B has no access to the container or Robot A's observations.",
     "The container is later moved to Robot B's room.",
     "A label attached to the container accurately states that it contains a blue cube."]
q = ["Does Robot B have enough information to identify the cube's color?"]
pair(6, "The Information Barrier", h + ["Robot B can read the label."] + q, h + ["Robot B cannot read the label."] + q, "Yes", "No")

h = ["A software system contains three modules: X, Y, and Z.", "Module X can operate only if Module Y is active.",
     "Module Y can operate independently."]
r = ["During testing, Module Y is active and Module Z is active.", "Module X has been disabled.",
     "Is the system configuration consistent with the stated dependencies?"]
pair(7, "The Hidden Dependency", h + ["Module Z can operate only if Module X is inactive."] + r,
     h + ["Module Z can operate only if Module X is active."] + r, "Yes", "No")

h = ["A university evaluates a new educational program.", "One hundred students participated in the program.",
     "Ninety students completed the program successfully.",
     "The university compares this result with an earlier group in which 70% of students completed a traditional program."]
q = ["Can the university conclude that the new program achieved a higher observed completion rate?"]
pair(8, "The Invisible Selection Bias",
     h + ["Students in both groups were randomly assigned, and both groups were evaluated using identical criteria."] + q,
     h + ["Students selected themselves into the new program, and the two groups were evaluated using different criteria."] + q,
     "Yes", "90% vs 70% is reported, but different criteria make the comparison potentially invalid", "open")

h = ["An automated decision system assigns priorities to incoming requests.",
     "Requests marked \"critical\" normally receive priority over requests marked \"standard.\"",
     "However, requests with verified safety implications override all other classifications.",
     "Request A is marked \"critical\" and has no verified safety implications.",
     "Request B is marked \"standard\" and has verified safety implications."]
q = ["Which request should be processed first?"]
pair(9, "The Priority Reversal", h + ["The override rule is mandatory."] + q,
     h + ["The override rule is advisory and may be ignored in favor of the normal classification system."] + q,
     "Request B", "Not uniquely determined (override is optional)", "open")

h = ["A logistics company must assign four deliveries to two vehicles.", "Vehicle A can carry a maximum of 600 kilograms.",
     "Vehicle B can carry a maximum of 500 kilograms.", "The deliveries weigh 200, 250, 300, and 350 kilograms.",
     "Each delivery must remain intact and must be assigned to exactly one vehicle.",
     "Both vehicles must carry at least one delivery."]
q = ["Is it possible to complete all four deliveries in a single trip?"]
pair(10, "The Final Constraint",
     h + ["The company is allowed to distribute the deliveries between the vehicles in any combination that respects their weight limits."] + q,
     h + ["The company is required to assign the two heaviest deliveries to Vehicle B."] + q,
     "Yes (A: 250+350, B: 200+300)", "No (350+300=650 > 500)")
json.dump(T, open("tests.json", "w"), indent=1)
print(len(T), "pairs")
