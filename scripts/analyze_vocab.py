#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_shannon_entropy(counts_per_doc, total_count):
    entropy = 0
    for count in counts_per_doc:
        p_i = count / total_count
        entropy -= p_i * math.log(p_i, 2)
    return entropy


def analyze_corpus(input_path: Path, output_csv: Path):
    logger.info(f"Analyzing: {input_path.name}")

    word_total_counts = Counter()
    word_doc_distributions = defaultdict(list)
    total_words = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if not words: continue

            total_words += len(words)
            line_counts = Counter(words)
            for word, count in line_counts.items():
                word_total_counts[word] += count
                word_doc_distributions[word].append(count)

    data = []
    for word, count in word_total_counts.items():
        entropy = calculate_shannon_entropy(word_doc_distributions[word], count)
        data.append({
            'lemma': word,
            'count': count,
            'relative_freq_10k': (count / total_words) * 10000,
            'entropy': round(entropy, 4),
            'doc_appearance': len(word_doc_distributions[word])
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    # ВАЖНО: Аргумента --plot здесь больше нет!
    args = parser.parse_args()

    analyze_corpus(Path(args.input), Path(args.output))