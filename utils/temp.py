from attribute_prompts import persona_selected, persona_prompts_3
import numpy as np

selected = [166,207,218,295,319,390,394]

info = {"note": "new attribute list", "user": "user1", "n": 200, "p": [-1.1945447871937713, 1.8237161056289326, 5.023001003590478, 1.5896923904881053, 1.5214860466689908, 1.4284395159296936, 1.4968375906615854], "lambda0": 0.01}

p = info["p"]
p = np.array(p)
p = p / np.linalg.norm(p)
p = p.tolist()

for x, y in zip(selected, p):
    print(f"({y:.4f}, {persona_prompts_3[x]})")

