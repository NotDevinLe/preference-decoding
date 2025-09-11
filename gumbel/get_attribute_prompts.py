#!/usr/bin/env python3
"""
Sample 1,000 unique personas from proj-persona/PersonaHub (config: 'reasoning').

Outputs:
  - personas_1000.jsonl  (one persona per line as {"persona": ...})
  - personas_1000.txt     (one persona per line)

Usage:
  python sample_personas.py --num 1000 --seed 0 --out-prefix personas_1000
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, concatenate_datasets

CANDIDATE_PERSONA_COLS = [
    "persona", "Persona", "persona_text", "persona_description",
    "profile", "character", "identity", "bio", "role"
]

def detect_persona_column(column_names: List[str]) -> Optional[str]:
    # 1) exact/partial match containing 'persona'
    for c in column_names:
        if "persona" in c.lower():
            return c
    # 2) conservative fallbacks
    for name in CANDIDATE_PERSONA_COLS:
        for c in column_names:
            if c.lower() == name.lower():
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="reasoning", help="HF dataset config (default: reasoning)")
    ap.add_argument("--dataset", default="proj-persona/PersonaHub", help="HF dataset path")
    ap.add_argument("--num", type=int, default=1000, help="How many unique personas to sample")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out-prefix", default="personas_1000", help="Output filename prefix")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load all available splits for the config
    ds_dict = load_dataset(args.dataset, args.config)
    splits = list(ds_dict.keys())

    if not splits:
        raise RuntimeError("No splits found for this dataset/config.")

    # Concatenate all splits (train/validation/test) if present
    if len(splits) == 1:
        ds_all = ds_dict[splits[0]]
    else:
        ds_all = concatenate_datasets([ds_dict[s] for s in splits])

    # Detect the persona column
    persona_col = detect_persona_column(ds_all.column_names)
    if persona_col is None:
        raise RuntimeError(
            f"Could not find a persona-like column in {ds_all.column_names}. "
            "Inspect the dataset to choose the correct field name."
        )

    # Collect unique personas
    uniques = set()
    for p in ds_all[persona_col]:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            uniques.add(s)

    uniques = list(uniques)
    total_unique = len(uniques)
    if total_unique == 0:
        raise RuntimeError(f"No non-empty personas found in column '{persona_col}'.")

    # Sample up to args.num (or all, if fewer than requested)
    k = min(args.num, total_unique)
    sampled = random.sample(uniques, k)

    # Write outputs
    out_jsonl = Path(f"{args.out_prefix}.jsonl")
    out_txt = Path(f"{args.out_prefix}.txt")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(json.dumps({"persona": s}, ensure_ascii=False) + "\n")

    with out_txt.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(s + "\n")

    print(f"[done] persona column: {persona_col}")
    print(f"[done] total unique personas found: {total_unique}")
    print(f"[done] sampled: {k}")
    print(f"[done] wrote: {out_jsonl} and {out_txt}")

if __name__ == "__main__":
    main()
