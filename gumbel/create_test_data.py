#!/usr/bin/env python3
"""
Create minimal test data for collector server testing.
"""

import pickle
import json
from pathlib import Path

def create_test_dataset():
    """Create minimal test dataset with 3 samples"""
    test_data = {
        'user1': {
            'prompts': [
                "What is the capital of France?",
                "How do you make pasta?", 
                "What is machine learning?"
            ],
            'outputs': [
                "The capital of France is Paris.",
                "Boil water, add pasta, cook for 8-10 minutes.",
                "Machine learning is a subset of AI that learns from data."
            ]
        }
    }
    
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("Created test_data.pkl with 3 samples")

def create_test_attribute_prompts():
    """Create minimal attribute prompts for testing"""
    test_prompts = {
        "prompts": [
            "You are a helpful assistant focused on accuracy",
            "You are a creative assistant focused on interesting responses", 
            "You are a concise assistant focused on brevity"
        ],
        "count": 3,
        "source": "test"
    }
    
    with open('test_attribute_prompts.json', 'w') as f:
        json.dump(test_prompts, f, indent=2)
    
    print("Created test_attribute_prompts.json with 3 attribute prompts")

if __name__ == "__main__":
    create_test_dataset()
    create_test_attribute_prompts()