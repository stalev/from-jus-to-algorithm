"""
Anchor Validation Script

This script checks whether anchor triplets (Latin–French–English)
exist in the corresponding corpus vocabularies and reports their
rank and frequency.

Expected project structure:

data/
    reference/
        alignment_anchors.json
    vocab/
        roman.csv
        napoleon.csv
        eu.csv

Each CSV file must contain the columns:
    - lemma
    - count

Run from the project root directory:

    python gadgets/check_anchors.py \
        --anchors data/reference/alignment_anchors.json \
        --vocab data/vocab

Make sure you execute the command from the root of the repository,
otherwise relative paths may not resolve correctly.
"""

import json
import pandas as pd
from pathlib import Path
import argparse  # Added for parameter support


def check_anchors(json_path, vocab_dir):
    # 1. Load the triplet "gold list" from JSON
    # Added a simple comment-stripping logic to handle your commented JSON files
    with open(json_path, 'r', encoding='utf-8') as f:
        content = "".join([line for line in f if not line.strip().startswith("//")])
        anchors = json.loads(content)

    # 2. Map language to file and key
    files = {
        "Latin (Roman)": {"file": "roman.csv", "key": "la"},
        "French (Napol)": {"file": "napoleon.csv", "key": "fr"},
        "English (EU)": {"file": "eu.csv", "key": "en"}
    }

    vocab_dir = Path(vocab_dir)

    # Table header
    print(f"{'TERM (LOCAL)':<20} | {'CORPUS':<15} | {'RANK':<6} | {'FREQ':<8} | {'STATUS'}")
    print("-" * 75)

    for item in anchors:
        # Iterate through each language for the current triplet
        for lang_name, info in files.items():
            path = vocab_dir / info['file']
            search_term = item[info['key']]  # Get translation for specific language

            if not path.exists():
                print(f"{search_term:<20} | {lang_name:<15} | {'ERR':<6} | {'-':<8} | ❌ NO CSV")
                continue

            # Read CSV (assuming sorted by frequency)
            df = pd.read_csv(path)

            # Find match in 'lemma' column
            match = df[df['lemma'] == search_term]

            if not match.empty:
                rank = match.index[0] + 1
                count = match['count'].values[0]
                status = "✅ FOUND"
            else:
                rank = "-"
                count = 0
                status = "❌ MISSING"

            print(f"{search_term:<20} | {lang_name:<15} | {rank:<6} | {count:<8} | {status}")

        # Separator between triplets
        print("-" * 75)


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Check anchor presence in vocabulary CSVs.")

    # Параметр называется --anchors, значит в коде это будет args.anchors
    parser.add_argument(
        "--anchors",
        type=str,
        default="data/reference/selected_anchors.json",
        help="Path to the JSON file with anchors"
    )

    parser.add_argument(
        "--vocab",
        type=str,
        default="results/vocab",
        help="Path to the directory with CSV vocabularies"
    )

    args = parser.parse_args()

    # ИСПРАВЛЕНО: используем args.anchors вместо args.file
    if Path(args.anchors).exists():
        check_anchors(args.anchors, args.vocab)
    else:
        print(f"Error: {args.anchors} not found!")