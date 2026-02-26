#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 — Semantic Displacement Ranking (COMPACT VISUALIZATION)
==========================================================
Identifies which legal concepts shifted the most across epochs.
Changes:
- Scatterplots have NO titles.
- Image sizes reduced by 50% (makes fonts look huge).
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec, FastText

# Optional: better label placement
try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("⚠️ install adjustText for better label placement (pip install adjustText)")

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
EPOCHS = ["roman", "napoleon", "eu"]
MODELS = ["w2v", "fasttext"]


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def load_model(epoch, model_type, models_dir):
    """Loads KeyedVectors from .model files"""
    path = models_dir / f"{epoch}_aligned_{model_type}.model"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    if model_type == "fasttext":
        model = FastText.load(str(path))
    else:
        model = Word2Vec.load(str(path))
    return model.wv


def cosine_distance(wv_a, word_a, wv_b, word_b):
    if word_a not in wv_a or word_b not in wv_b:
        return None
    return cosine(wv_a[word_a], wv_b[word_b])


def parse_anchors(json_path):
    """Parses list-based JSON anchors"""
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    term_map = {}
    for item in raw_data:
        concept = item.get("en", "unknown")
        term_roman = item.get("la")
        term_napoleon = item.get("fr")
        term_eu = item.get("en")

        if term_roman and term_napoleon and term_eu:
            term_map[concept] = [term_roman, term_napoleon, term_eu]
    return term_map


# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
def plot_displacement_bar(df, model_type, out_dir):
    df_sorted = df.sort_values("Rome_to_EU", ascending=False)

    # REDUCED SIZE: Was (10, 12) -> Now (5, 6)
    plt.figure(figsize=(5, 6))

    norm = plt.Normalize(df_sorted["Rome_to_EU"].min(), df_sorted["Rome_to_EU"].max())
    colors = plt.cm.RdYlGn_r(norm(df_sorted["Rome_to_EU"].values))

    sns.barplot(
        x="Rome_to_EU",
        y="concept",
        data=df_sorted,
        palette=list(colors),
        hue="concept",
        legend=False
    )

    # Keep title for Bar Chart
    plt.title(f"Displacement (Rome → EU)\n{model_type.upper()}", fontsize=14)
    plt.xlabel("Cosine Distance", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    out_bar = out_dir / f"displacement_bar_{model_type}.png"
    plt.savefig(out_bar, dpi=150)
    plt.close()


def plot_displacement_scatter(df, model_type, out_dir):
    # REDUCED SIZE: Was (16, 14) -> Now (8, 7)
    plt.figure(figsize=(8, 7))

    # Scatter plot
    scatter = sns.scatterplot(
        data=df,
        x="Rome_to_Nap",
        y="Nap_to_EU",
        hue="Rome_to_EU",
        palette="viridis",
        s=600,  # Slightly reduced bubble size for smaller canvas
        edgecolor="black",
        linewidth=1.5
    )

    # Labels
    texts = []
    for _, r in df.iterrows():
        texts.append(plt.text(r["Rome_to_Nap"], r["Nap_to_EU"], r["concept"],
                              fontsize=14, weight="bold"))  # Font looks huge on 8x7

    if HAS_ADJUST_TEXT:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=1))

    # Reference lines
    plt.axvline(df["Rome_to_Nap"].mean(), ls="--", c="grey", alpha=0.5)
    plt.axhline(df["Nap_to_EU"].mean(), ls="--", c="grey", alpha=0.5)

    # NO TITLE HERE as requested

    plt.xlabel("Rome → Napoleon", fontsize=12)
    plt.ylabel("Napoleon → EU", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Compact Legend
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(
        handles=handles[::-1],
        labels=labels[::-1],
        title="Total Shift",
        fontsize=10,
        title_fontsize=11,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        framealpha=0.9
    )

    plt.tight_layout()

    out_scatter = out_dir / f"displacement_scatter_{model_type}.png"
    plt.savefig(out_scatter, dpi=150)
    plt.close()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--fig_dir", required=True)
    parser.add_argument("--anchors", required=True)
    args = parser.parse_args()

    m_dir = Path(args.models_dir)
    met_dir = Path(args.metrics_dir)
    f_dir = Path(args.fig_dir)

    try:
        term_map = parse_anchors(args.anchors)
        print(f"Loaded {len(term_map)} anchors.")
    except Exception as e:
        print(f"Error parsing anchors: {e}");
        return

    for m_type in MODELS:
        print(f"\nProcessing {m_type.upper()}...")
        try:
            wvs = {ep: load_model(ep, m_type, m_dir) for ep in EPOCHS}
        except Exception as e:
            print(f"Skipping {m_type}: {e}");
            continue

        rows = []
        for concept, terms in term_map.items():
            w_r, w_n, w_e = terms[0], terms[1], terms[2]

            d_rn = cosine_distance(wvs["roman"], w_r, wvs["napoleon"], w_n)
            d_ne = cosine_distance(wvs["napoleon"], w_n, wvs["eu"], w_e)
            d_re = cosine_distance(wvs["roman"], w_r, wvs["eu"], w_e)

            if None not in (d_rn, d_ne, d_re):
                rows.append({
                    "concept": concept.replace("_", " ").title(),
                    "terms": f"{w_r}/{w_n}/{w_e}",
                    "Rome_to_Nap": d_rn,
                    "Nap_to_EU": d_ne,
                    "Rome_to_EU": d_re
                })

        if not rows: continue

        df = pd.DataFrame(rows)
        df.to_csv(met_dir / f"displacement_ranking_{m_type}.csv", index=False)

        plot_displacement_bar(df, m_type, f_dir)
        plot_displacement_scatter(df, m_type, f_dir)
        print(f"   Saved plots to {f_dir}")


if __name__ == "__main__":
    main()