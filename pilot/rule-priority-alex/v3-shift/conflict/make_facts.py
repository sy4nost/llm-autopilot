"""Own facts for the conflict lamp: each fact in a 'consistent' version (passage = truth) and a 'conflict' version
(passage swaps in another answer of the same type). Candidates = [passage answer, memory answer] (same pair for both versions).
  python3 -I make_facts.py  -> facts.json"""
import json, random
CAP = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"),
       ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Brasilia"),
       ("Argentina", "Buenos Aires"), ("Kenya", "Nairobi"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Norway", "Oslo"),
       ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
       ("Ireland", "Dublin"), ("Turkey", "Ankara"), ("India", "New Delhi"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
       ("Peru", "Lima"), ("Chile", "Santiago"), ("Mexico", "Mexico City"), ("Cuba", "Havana"), ("Iran", "Tehran"),
       ("Iraq", "Baghdad"), ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
       ("Czech Republic", "Prague"), ("Ukraine", "Kyiv"), ("South Korea", "Seoul"), ("Indonesia", "Jakarta"), ("Colombia", "Bogota")]
EL = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("sodium", "Na"), ("potassium", "K"), ("lead", "Pb"), ("copper", "Cu"),
      ("tin", "Sn"), ("mercury", "Hg"), ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"), ("nitrogen", "N"), ("helium", "He"),
      ("calcium", "Ca"), ("zinc", "Zn"), ("chlorine", "Cl"), ("neon", "Ne"), ("sulfur", "S"), ("uranium", "U")]
AU = [("Hamlet", "Shakespeare"), ("War and Peace", "Tolstoy"), ("Crime and Punishment", "Dostoevsky"), ("1984", "Orwell"),
      ("Don Quixote", "Cervantes"), ("The Odyssey", "Homer"), ("Pride and Prejudice", "Austen"), ("Faust", "Goethe"),
      ("The Divine Comedy", "Dante"), ("Oliver Twist", "Dickens"), ("Moby-Dick", "Melville"), ("Les Miserables", "Hugo"),
      ("The Old Man and the Sea", "Hemingway"), ("Ulysses", "Joyce"), ("The Trial", "Kafka"), ("Eugene Onegin", "Pushkin"),
      ("The Seagull", "Chekhov"), ("Frankenstein", "Shelley"), ("Dracula", "Stoker"), ("The Little Prince", "Saint-Exupery")]
MISC = [("the largest planet in the Solar System", "Jupiter"), ("the planet closest to the Sun", "Mercury"),
        ("the longest river in Africa", "the Nile"), ("the largest ocean", "the Pacific"), ("the tallest mountain on Earth", "Everest"),
        ("the largest desert in Africa", "the Sahara"), ("the inventor of the telephone", "Bell"), ("the painter of the Mona Lisa", "da Vinci"),
        ("the composer of the Moonlight Sonata", "Beethoven"), ("the discoverer of penicillin", "Fleming"),
        ("the first person to walk on the Moon", "Armstrong"), ("the author of the theory of relativity", "Einstein"),
        ("the hardest natural mineral", "diamond"), ("the largest mammal", "the blue whale"), ("the fastest land animal", "the cheetah"),
        ("the currency of Japan", "the yen"), ("the currency of the United Kingdom", "the pound"), ("the official language of Brazil", "Portuguese"),
        ("the smallest prime number", "2"), ("the number of continents", "7")]
T = {"cap": ("Travel guide, page 12. Many visitors start their trip in the capital. The capital of {s} is {a}, a city known for its museums and markets.",
             "What is the capital of {s}?"),
     "el": ("Chemistry handout. Each element has a short symbol used in formulas. The chemical symbol for {s} is {a}.",
            "What is the chemical symbol for {s}?"),
     "au": ("Library catalogue note. The novel or play {s} was written by {a}, and it is often studied in schools.",
            "Who wrote {s}?"),
     "misc": ("Quiz notes for the school team. Remember this one: {s} is {a}.", "What is {s}?")}
rng = random.Random(0)
out = []
for kind, rows in (("cap", CAP), ("el", EL), ("au", AU), ("misc", MISC)):
    for i, (s, true) in enumerate(rows):
        others = [r[1] for j, r in enumerate(rows) if j != i]
        fake = rng.choice(others)
        p, q = T[kind]
        for version, ans in (("consistent", true), ("conflict", fake)):
            other = fake if version == "consistent" else true
            out.append(dict(id=f"{kind}{i}", kind=kind, version=version, prompt=f"Passage: {p.format(s=s, a=ans)}\nQuestion: {q.format(s=s)}\nAnswer with just the answer.",
                            cands=[ans, other], gold=0, memory=1 if version == "conflict" else 0))
json.dump(out, open("facts.json", "w"), indent=0)
print(len(out), "items")
