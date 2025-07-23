import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file")
args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f)

for item in data:
    if "instruction" in item:
        item["prompt"] = item.pop("instruction")

with open(args.output, "w") as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} items. Saved to {args.output}")
