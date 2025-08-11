#!/usr/bin/env python3
"""
Consolidate individual BON output files into a single file for run_generation.py
"""

import json
import glob
from pathlib import Path

def consolidate_bon_files(input_dir: str, output_file: str):
    """Consolidate all prompt_XXXX_outputs.json files into a single file."""
    input_path = Path(input_dir)
    
    # Find all prompt files
    prompt_files = sorted(input_path.glob("prompt_*_outputs.json"))
    print(f"Found {len(prompt_files)} prompt files")
    
    consolidated_data = []
    
    for file_path in prompt_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create entry in the format expected by run_generation.py
        entry = {
            "prompt": data["prompt"],
            "outputs": data["outputs"]
        }
        
        consolidated_data.append(entry)
    
    # Save consolidated file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    print(f"Consolidated {len(consolidated_data)} prompts into {output_file}")

if __name__ == "__main__":
    consolidate_bon_files("results/bon_outputs", "results/consolidated_bon_data.json")