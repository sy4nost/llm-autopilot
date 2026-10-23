"""12 domains x 5 subtopics = 60 host texts for the odd-phrase pilot."""
DOMAINS = {
    "cooking": ["baking sourdough bread", "making fresh pasta", "fermenting vegetables", "grilling fish", "tempering chocolate"],
    "sailing": ["trimming the sails of a sloop", "navigating with a paper chart", "anchoring in a crowded bay", "tides and tidal currents", "maintaining a wooden hull"],
    "astronomy": ["the phases of the Moon", "red giant stars", "the rings of Saturn", "observing with a backyard telescope", "comets and their tails"],
    "gardening": ["growing tomatoes", "composting kitchen waste", "pruning fruit trees", "watering a vegetable bed", "starting seedlings indoors"],
    "medieval history": ["life in a medieval castle", "medieval craft guilds", "the crusades", "monasteries in the Middle Ages", "medieval farming"],
    "computer networks": ["how routers forward packets", "Wi-Fi signal strength", "the domain name system", "network firewalls", "fiber optic cables"],
    "football": ["the offside rule", "training a goalkeeper", "penalty shootouts", "a football club youth academy", "tactical formations"],
    "geology": ["how volcanoes form", "sedimentary rocks", "earthquakes and fault lines", "caves and stalactites", "glaciers shaping valleys"],
    "musical instruments": ["how a violin produces sound", "tuning a piano", "learning the drums", "the pipe organ", "playing the trumpet"],
    "banking": ["how savings accounts earn interest", "applying for a mortgage", "credit card fees", "central bank interest rates", "opening a business bank account"],
    "insects": ["honeybee colonies", "ant foraging trails", "butterfly metamorphosis", "dragonfly hunting", "fireflies and their light"],
    "railways": ["steam locomotives", "high-speed trains", "railway signalling", "laying railway track", "freight trains and wagons"],
}
TOPICS = [(d, s) for d, subs in DOMAINS.items() for s in subs]
