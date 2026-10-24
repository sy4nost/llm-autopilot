"""three-forces/base: same items as ../sets_tf.json, rendered as a plain-text transcript so base (pretrained-only) and
instruct models see the identical prompt.  python3 make_base.py -> sets_base.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
items = json.load(open(os.path.join(HERE, "..", "sets_tf.json")))["items"]
HEAD = "The following is a conversation between a user and a helpful assistant.\n\n"


def render(msgs):
    lines = [("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in msgs if m["role"] != "system"]
    return HEAD + "\n".join(lines) + "\nAssistant:"


for it in items:
    it["text"] = render(it["messages"])
json.dump(dict(items=items), open(os.path.join(HERE, "sets_base.json"), "w"))
print(len(items), "items"); print(items[5]["text"])
