"""Stages 3 and 4 of Alex's benchmark.
Stage 3 (tests_rephrased.json): the same 10 logical structures, new story, names, numbers and wording (written by Claude).
Stage 4 (tests_distractor.json): Alex's original A/B with two irrelevant sentences inserted (they change no answer).
"""
import json
R = []
def pair(n, name, a, b, ka, kb, kind="yesno"):
    R.append({"test": n, "name": name, "A": "\n".join(a), "B": "\n".join(b), "key_A": ka, "key_B": kb, "kind": kind})

h = ["In a hospital pharmacy, every change to a prescription must be signed off by both the attending physician and the duty pharmacist before the medicine is dispensed."]
t = ["A routine refill has been signed off by the attending physician. The duty pharmacist has not checked it.",
     "May the refill be dispensed under the pharmacy's rules?"]
pair(1, "Silent Override (rephrased)", h + ["Routine refills may skip the duty pharmacist's check."] + t,
     h + ["Routine refills may skip the attending physician's sign-off."] + t, "Yes", "No")

h = ["A vibration sensor in a pumping station writes events to a log.", "A booster pump starts automatically whenever the main water valve closes.",
     "The log shows a burst of vibration with the timestamp 14:30.", "The main water valve closed at 14:27."]
q = ["Could the logged vibration have been caused by the booster pump starting after the valve closed?"]
pair(2, "Delayed Signal (rephrased)", h + ["Every log entry is timestamped exactly five minutes after the event it describes."] + q,
     h + ["Every log entry is timestamped exactly two minutes after the event it describes."] + q,
     "No (event 14:25, before 14:27)", "Yes (event 14:28, after 14:27)")

h = ["A bank approves small business loans using an internal scoring rule.", "An applicant qualifies if the credit score is at least 700.",
     "Applicant Pereira has a score of 742.", "Applicant Quist has a score of 688.", "Qualified applicants receive the loan."]
q = ["Which applicants receive the loan?"]
pair(3, "Definition Shift, control (rephrased)", h + ["A score of exactly 700 counts as qualifying."] + q,
     h + ["A score of exactly 700 does not count; the score must be strictly above 700."] + q, "Pereira only", "Pereira only", "open")

h = ["Five engineers, Farid, Grace, Hugo, Ines and Jonah, can work the night shift.", "Exactly three engineers must work the night shift.",
     "Grace must work whenever Hugo works.", "Farid cannot work if Ines works.", "Hugo and Ines are already on the shift."]
q = ["Can a valid night shift be formed?"]
pair(4, "Missing Participant (rephrased)", h + ["Jonah is on leave."] + q, h + ["Grace is on leave."] + q,
     "Yes (Hugo, Ines, Grace)", "No (Hugo requires Grace)")

h = ["A server room is supplied by two independent power feeds."]
r = ["During maintenance, Feed 1 was cut.", "Feed 2 kept working normally.", "The servers kept running.",
     "If Feed 1 had not been cut and everything else had stayed the same, would the servers still have kept running?"]
pair(5, "Unchanged Outcome (rephrased)", h + ["Either feed on its own is enough to keep the servers running."] + r,
     h + ["Both feeds must be active at the same time to keep the servers running."] + r,
     "Yes", "Yes, but the data contradict the rule (Feed 2 alone kept them running)", "open")

h = ["Two delivery drones work in different warehouses and share no data.", "Drone A seals a red package inside an opaque box.",
     "Drone B has no access to the box or to Drone A's records.", "The box is later handed over to Drone B.",
     "A printed tag on the box correctly says that it contains a red package."]
q = ["Does Drone B have enough information to know the package's color?"]
pair(6, "Information Barrier (rephrased)", h + ["Drone B's scanner can read the tag."] + q,
     h + ["Drone B's scanner is broken and cannot read the tag."] + q, "Yes", "No")

h = ["A water system has three valves: P, Q and R.", "Valve P can be open only if Valve Q is open.", "Valve Q can be opened independently."]
r = ["During an inspection, Valve Q is open and Valve R is open.", "Valve P is closed.",
     "Is this state consistent with the stated rules?"]
pair(7, "Hidden Dependency (rephrased)", h + ["Valve R can be open only if Valve P is closed."] + r,
     h + ["Valve R can be open only if Valve P is open."] + r, "Yes", "No")

h = ["A clinic evaluates a new physiotherapy protocol.", "Eighty patients followed the new protocol.",
     "Sixty of them recovered within six weeks.",
     "The clinic compares this with an earlier group in which 50% of patients recovered within six weeks on the standard protocol."]
q = ["Can the clinic conclude that the new protocol had a higher observed recovery rate?"]
pair(8, "Selection Bias (rephrased)",
     h + ["Patients were randomly assigned to the two protocols, and recovery was judged by the same criteria in both groups."] + q,
     h + ["Patients chose the new protocol themselves, and recovery was judged by different criteria in the two groups."] + q,
     "Yes", "75% vs 50% is reported, but different criteria make the comparison potentially invalid", "open")

h = ["A help desk system orders incoming tickets.", "Tickets marked \"urgent\" are normally handled before tickets marked \"normal\".",
     "However, tickets that affect payments override every other label.",
     "Ticket X is marked \"urgent\" and does not affect payments.", "Ticket Y is marked \"normal\" and is confirmed to affect payments."]
q = ["Which ticket should be handled first?"]
pair(9, "Priority Reversal (rephrased)", h + ["The payment override must always be applied."] + q,
     h + ["The payment override is only a recommendation and staff may ignore it and use the normal labels instead."] + q,
     "Ticket Y", "Not uniquely determined (override is optional)", "open")

h = ["A ferry service must load four crates onto two boats.", "Boat A can carry at most 700 kilograms.",
     "Boat B can carry at most 450 kilograms.", "The crates weigh 150, 200, 300 and 400 kilograms.",
     "Each crate must go whole onto exactly one boat.", "Each boat must carry at least one crate."]
q = ["Can all four crates be moved in a single crossing?"]
pair(10, "Final Constraint (rephrased)",
     h + ["The crates may be split between the boats in any way that respects the weight limits."] + q,
     h + ["The two heaviest crates must go on Boat B."] + q,
     "Yes (A: 300+400=700, B: 150+200=350)", "No (300+400=700 > 450)")
json.dump(R, open("tests_rephrased.json", "w"), indent=1)

# stage 4: Alex's originals + two irrelevant sentences after the first line of each version
D = {1: ["The laboratory was founded in 1998 and currently employs forty-two researchers.",
         "Last spring, the safety officer completed a two-week certification course in chemical handling."],
     2: ["The facility is located about forty kilometers from the nearest town.",
         "The monitoring software was upgraded to a new version in March."],
     3: ["The company was founded in 2011 and has offices in three countries.",
         "Both systems were developed by the same engineering team over eighteen months."],
     4: ["The project is funded by a regional science foundation.",
         "Clara and Emma previously co-authored a paper on soil chemistry."],
     5: ["The plant produces industrial ceramics and runs three shifts a day.",
         "Cooling System B was installed two years after Cooling System A."],
     6: ["The robots were built by different manufacturers.",
         "Robot B is also responsible for sorting incoming mail in its room."],
     7: ["The system was written in a mix of Python and C++.",
         "Module Y also writes a short status message to a log file every hour."],
     8: ["The university has offered courses in this subject for over thirty years.",
         "The new program uses online materials and weekly group discussions."],
     9: ["The system processes about two thousand requests per day.",
         "Request A was submitted at 10:02, and Request B was submitted at 10:05."],
     10: ["Vehicle A is painted blue, and Vehicle B is painted white.",
          "The company has been operating in the region for twelve years."]}
orig = json.load(open("tests.json"))
S4 = []
for t in orig:
    nt = dict(t, name=t["name"] + " (with distractors)")
    for v in ("A", "B"):
        lines = t[v].split("\n")
        nt[v] = "\n".join(lines[:1] + D[t["test"]] + lines[1:])
    S4.append(nt)
json.dump(S4, open("tests_distractor.json", "w"), indent=1)
print(len(R), len(S4))
