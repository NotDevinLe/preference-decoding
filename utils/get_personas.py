import json
from datasets import load_dataset
import random
ds = load_dataset("SynthLabsAI/PERSONA")
ds = ds["train"]

personas = set()
for row in ds:
    personas.add(row["persona"])

personas = list(personas)
random.shuffle(personas)

train_personas = personas[:120]
test_personas = personas[120:150]

with open("configs/persona_prompts.json", "w") as f:
    json.dump({"train": train_personas, "test": test_personas}, f, indent=2)