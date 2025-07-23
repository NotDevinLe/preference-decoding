import pickle
import json

for i in range(1, 18):
    with open(f"../../data/user{i}.pkl", "rb") as f:
        data = pickle.load(f)
        with open("../../data/user_prompts.jsonl", "a") as f:
            f.write(json.dumps({"user": i, "prompt": data['user']}) + "\n")