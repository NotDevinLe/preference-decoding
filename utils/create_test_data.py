#!/usr/bin/env python3
"""
Create minimal test data for collector server testing.
"""

import pickle
import json
from pathlib import Path

def create_test_dataset():
    """Create interpretable test dataset with clear persona matches"""
    test_data = {
        'user1': [
            {
                'prompt': "Tell me about treasure hunting",
                'output': "Arrr, matey! Treasure hunting be the finest adventure on the seven seas! Ye'll need a trusty map, a sharp cutlass, and nerves of steel to find buried gold!"
            },
            {
                'prompt': "Tell me about treasure hunting",
                'output': "Treasure hunting is an archaeological practice involving the systematic search for valuable historical artifacts using proper documentation and preservation methods."
            },
            {
                'prompt': "Tell me about treasure hunting", 
                'output': "OMG treasure hunting is sooo cool! Like, you get to dig around and find shiny things! It's like the ultimate shopping spree but underground! 💎✨"
            }
        ]
    }
    
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("Created test_data.pkl with 3 samples")

def create_test_attribute_prompts():
    """Create interpretable attribute prompts that clearly match the test responses"""
    test_prompts = {
        "prompts": [
            "You are a pirate who speaks with nautical slang and says 'arrr'",
            "You are a formal academic scholar who uses precise technical language",
            "You are an enthusiastic teenager who uses slang and emojis"
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