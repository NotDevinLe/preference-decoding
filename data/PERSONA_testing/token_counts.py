"""Token length histogram for PERSONA testing data.

Scans JSON files in the dataset directory, computes token counts for each
chosen/rejected text using the Llama 3.2 1B tokenizer, and saves a
histogram plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def iter_texts(files: Iterable[Path]) -> Iterator[str]:
    """Yield all relevant text fields from each JSON file."""

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {file_path}: {exc}") from exc

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            for key in ("chosen", "rejected"):
                text = item.get(key)
                if text:
                    yield text


def compute_token_counts(tokenizer, texts: Iterable[str]) -> List[int]:
    counts: List[int] = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        counts.append(len(tokens))
    return counts


def plot_histogram(counts: List[int], output_path: Path) -> None:
    if not counts:
        raise RuntimeError("No token counts computed; check dataset contents")

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, color="steelblue", edgecolor="black")
    plt.title("Token Count Distribution")
    plt.xlabel("Token count")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot token count histogram")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing persona JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "token_count_hist.png",
        help="Path to save the histogram plot",
    )
    args = parser.parse_args()

    json_files = sorted(args.data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = list(iter_texts(json_files))
    counts = compute_token_counts(tokenizer, texts)
    plot_histogram(counts, args.output)
    print(f"Histogram saved to {args.output}")


if __name__ == "__main__":
    main()

