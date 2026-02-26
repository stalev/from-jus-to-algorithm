#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging

# Check for adjustText library to prevent label overlapping
try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# Configure logging for better traceability in scientific workflows
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_plot(input_csv: Path, output_plot: Path, anchors_json: Path, min_count: int = 5):
    """
    Generates a Frequency-Entropy scatter plot and labels anchor terms with IDs.
    """
    logger.info(f"Loading vocabulary data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Filter rare terms to reduce noise
    plot_df = df[df['count'] >= min_count].copy()
    if plot_df.empty:
        logger.warning("Not enough data to generate plot after filtering.")
        return

    # Load cross-lingual stable anchors
    if not anchors_json.exists():
        logger.error(f"Anchor file {anchors_json} not found!")
        return

    with open(anchors_json, 'r', encoding='utf-8') as f:
        anchors = json.load(f)

    # Determine the language key based on the filename to match the correct column in JSON
    # 'roman' -> Latin (la), 'napoleon' -> French (fr), 'eu' -> English (en)
    input_filename = str(input_csv).lower()
    if 'napoleon' in input_filename:
        lang_key = 'fr'
    elif 'eu' in input_filename:
        lang_key = 'en'
    else:
        lang_key = 'la'

    logger.info(f"Using language key '{lang_key}' for anchor mapping.")

    # Prepare mapping and legend entries
    legend_entries = []
    anchor_map = {}  # target_lemma -> ID
    for i, a in enumerate(anchors, 1):
        target_lemma = a[lang_key]
        anchor_map[target_lemma] = i
        legend_entries.append(f"{i:2d} | {a['la']:15} | {a['fr']:15} | {a['en']}")

    # Setup plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # 1. Background cloud: all terms in the corpus
    sns.scatterplot(
        data=plot_df,
        x='count',
        y='entropy',
        alpha=0.2,
        color='gray',
        edgecolor=None,
        s=15
    )

    # 2. Highlighted set: anchor terms found in the current corpus
    texts = []
    anchors_in_df = plot_df[plot_df['lemma'].isin(anchor_map.keys())].copy()

    sns.scatterplot(
        data=anchors_in_df,
        x='count',
        y='entropy',
        color='red',
        s=80,
        label=f'Alignment Anchors (Language: {lang_key})',
        zorder=5
    )

    # Add numeric IDs to the plot
    for _, row in anchors_in_df.iterrows():
        num = anchor_map[row['lemma']]
        texts.append(plt.text(
            row['count'],
            row['entropy'],
            str(num),
            fontsize=11,
            fontweight='bold',
            zorder=6
        ))

    # Axis configuration
    plt.xscale('log')
    plt.title(f'Semantic Stability Analysis: Anchor Distribution ({lang_key.upper()})', fontsize=20, pad=20)
    plt.xlabel('Frequency (Log Scale)', fontsize=14)
    plt.ylabel('Shannon Entropy', fontsize=14)
    plt.legend(loc='upper left', frameon=True)

    # Optimize label placement to avoid overlapping
    if HAS_ADJUST_TEXT and texts:
        logger.info("Optimizing label positions...")
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.5)
        )

    # Save the resulting figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Save the textual legend for reference in the paper
    legend_file = output_plot.parent / f"{input_csv.stem}_anchors_legend.txt"
    with open(legend_file, 'w', encoding='utf-8') as f:
        f.write("ID | LATIN           | FRENCH          | ENGLISH\n")
        f.write("-" * 55 + "\n")
        f.write("\n".join(legend_entries))

    logger.info(f"Plot saved to: {output_plot}")
    logger.info(f"Legend saved to: {legend_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot anchor distribution in Frequency-Entropy space.")
    parser.add_argument("--input", required=True, help="Path to vocabulary CSV")
    parser.add_argument("--output", required=True, help="Path for the output plot")
    parser.add_argument("--anchors", default="data/reference/alignment_anchors.json", help="Path to anchors JSON")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum frequency threshold")
    args = parser.parse_args()
    generate_plot(Path(args.input), Path(args.output), Path(args.anchors), args.min_count)