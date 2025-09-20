import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from glob import glob

def load_data(data_dir: str) -> Dict:

    """
    Loads data from a directory into a dictionary.

    Args:
        data_dir: Directory containing the data

    Returns:
        List of dictionaries in the form of [{"prompt": str, "chosen": str, "rejected": str}]
    """
    data = []
    for file in os.listdir(data_dir):
        if not (file.endswith("train.json") or file.endswith("test.json") or file.endswith("val.json")):
            continue
        with open(os.path.join(data_dir, file), 'r') as f:
            full_data = json.load(f)
            data.extend(full_data)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_file = args.output_file

    data = load_data(data_dir)

    

if __name__ == "__main__":
    main()