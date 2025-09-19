from attribute_prompts import persona_prompts
import json

data = {
    "prompts": persona_prompts
}

with open("../gumbel/configs/attribute_prompts_400.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
