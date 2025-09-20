#!/usr/bin/env python3
"""
Convert personas.jsonl to the JSON format expected by collector_server.py
"""

import json
import argparse
from pathlib import Path

def convert_jsonl_to_json(input_path: str, output_path: str):
    """Convert personas.jsonl to JSON format for collector server"""
    
    personas = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                personas.append(data['persona'])
    
    # Save in format expected by collector_server.py
    output_data = {
        "prompts": personas,
        "count": len(personas),
        "source": "PersonaHub"
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Converted {len(personas)} personas from {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert personas.jsonl to JSON format")
    parser.add_argument("--input", default="personas_1000.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="attribute_prompts.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return 1
    
    convert_jsonl_to_json(args.input, args.output)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())