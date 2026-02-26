#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lemma Extraction Script
=======================
Input: Normalized JSON legal corpus
Output: Plain .txt file containing only lemmas (one paragraph per line)
Purpose: Prepare data for Word2Vec/FastText training
"""

import json
import argparse
import logging
from pathlib import Path

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_lemmas(input_json: Path, output_txt: Path):
    """
    Reads the nested JSON structure and extracts 'lemmatized_text'
    from each paragraph.
    """
    logger.info(f"Extracting lemmas from {input_json.name}...")

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lemmas_list = []

    # Traverse the nested structure: parts -> books -> titles -> paragraphs
    for part in data.get("parts", []):
        for book in part.get("books", []):
            # 1. Collect paragraphs belonging directly to the book
            paragraphs = book.get("paragraphs", [])

            # 2. Collect paragraphs nested within titles
            for title in book.get("titles", []):
                paragraphs.extend(title.get("paragraphs", []))

            # 3. Extract the actual lemma strings
            for p in paragraphs:
                text = p.get("lemmatized_text", "").strip()
                if text:
                    lemmas_list.append(text)

    # Save results: each paragraph on a new line
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lemmas_list))

    logger.info(f"Successfully saved {len(lemmas_list)} paragraphs to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lemmas from JSON to TXT")
    parser.add_argument("--input", required=True, help="Path to normalized JSON file")
    parser.add_argument("--output", required=True, help="Path to output .txt file")
    args = parser.parse_args()

    extract_lemmas(Path(args.input), Path(args.output))