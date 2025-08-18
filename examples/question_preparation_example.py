#!/usr/bin/env python3
"""
Example of how to use the question preparation script.
"""

import subprocess
import json
from pathlib import Path

def run_preparation_example():
    """Show how to prepare questions from different sources."""
    
    print("="*60)
    print("QUESTION PREPARATION EXAMPLES")
    print("="*60)
    
    # Example 1: Prepare questions from Dolly dataset
    print("\n1. Preparing questions from Dolly dataset...")
    print("-" * 40)
    
    cmd = [
        "python", "scripts/generate/prepare_questions.py",
        "--source", "dolly",
        "--output-file", "data/dolly_questions.json",
        "--num-questions", "100",
        "--seed", "42"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Success!")
        
        # Show the prepared questions
        with open("data/dolly_questions.json", 'r') as f:
            data = json.load(f)
        
        print(f"Prepared {data['metadata']['num_questions']} questions")
        print("Sample questions:")
        for i, q in enumerate(data['questions'][:3]):
            print(f"  {i+1}. {q[:80]}...")
    else:
        print("❌ Failed:")
        print(result.stderr)

if __name__ == "__main__":
    run_preparation_example()