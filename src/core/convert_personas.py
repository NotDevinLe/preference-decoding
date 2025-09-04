import json
from pathlib import Path

INPUT_PATH = Path("src/core/sampled_personas.json")
OUTPUT_PATH = Path("src/core/personas_only.json")

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    personas = []
    # Expected structure: { "dataset_info": {...}, "personas": [{"row_index": int, "persona": str}, ...] }
    items = data.get("personas", [])
    for item in items:
        persona_text = item.get("persona")
        if isinstance(persona_text, str):
            personas.append(persona_text)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(personas)} personas to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
