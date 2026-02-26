#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B3 — Technicality Evolution Analysis (COMPACT HEATMAPS)
=======================================================
1. Reads anchors from list-based JSON.
2. Calculates Semantic Density.
3. Generates Visualizations with REDUCED CELL SIZE for Heatmaps.
"""

import pandas as pd
import argparse
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
from gensim.models import Word2Vec, FastText
from scipy.stats import pearsonr

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
EPOCHS = ["roman", "napoleon", "eu"]
MODELS = ["w2v", "fasttext"]
TOP_K = 25

# FIXED PALETTE
HIGHLIGHTS = {
    "obligation": "#1f77b4",  # Blue
    "contract": "#d62728",  # Red
    "consent": "#2ca02c",  # Green
    "condition": "#ff7f0e",  # Orange
    "performance": "#9467bd",  # Purple
    "agreement": "#8c564b",  # Brown
    "action": "#e377c2",  # Pink
    "damage": "#7f7f7f",  # Gray
    "remedy": "#bcbd22",  # Olive
    "price": "#17becf",  # Cyan
    "exception": "#d6616b"  # Light Red
}


# ------------------------------------------------------------------
# LOGIC: DATA LOADING
# ------------------------------------------------------------------
def parse_anchors(json_path):
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


def load_model(epoch, m_type, m_dir):
    path = m_dir / f"{epoch}_aligned_{m_type}.model"
    if not path.exists(): raise FileNotFoundError(f"Missing model: {path}")
    load_func = FastText.load if m_type == "fasttext" else Word2Vec.load
    return load_func(str(path)).wv


def get_density(wv, word, k=TOP_K):
    if word not in wv: return None
    neighbors = wv.most_similar(word, topn=k)
    return np.mean([sim for _, sim in neighbors])


# ------------------------------------------------------------------
# LOGIC: VISUALIZATION
# ------------------------------------------------------------------
def adjust_labels_y(labels, min_dist=0.03):
    labels.sort(key=lambda x: x['y'])
    for i in range(1, len(labels)):
        prev = labels[i - 1]
        curr = labels[i]
        dist = curr['y'] - prev['y']
        if dist < min_dist:
            shift = (min_dist - dist)
            curr['y'] += shift
    return labels


def plot_heatmap(df, m_type, out_dir):
    """
    Individual Heatmap with reduced size (Height / 3, Width / 2).
    """
    pivot = df.pivot(index="concept", columns="epoch", values="density")[EPOCHS]

    if "roman" in pivot.columns and "eu" in pivot.columns:
        pivot["delta"] = pivot["eu"] - pivot["roman"]
        pivot = pivot.sort_values("delta", ascending=True)
        pivot = pivot.drop(columns=["delta"])

    pivot.columns = ["Roman", "Napoleon", "EU"]

    # REDUCED SIZE: Was (10, 14) -> Now (5, 4.7)
    plt.figure(figsize=(5, 4.7))

    ax = sns.heatmap(
        pivot,
        cmap="RdYlBu",
        annot=True,
        fmt=".2f",
        linewidths=.5,
        annot_kws={'size': 14},
        cbar_kws={'label': 'Semantic Density'}
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Semantic Density', size=16)

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.title(f"Semantic Erosion ({m_type.upper()})", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_erosion_{m_type}.png", dpi=200)
    plt.close()
    print(f"   🖼️  Saved Heatmap      -> heatmap_erosion_{m_type}.png")


def plot_combined_heatmap(all_dfs, out_dir):
    """
    Side-by-Side Heatmap with reduced size (Height / 3, Width / 2).
    """
    if 'w2v' not in all_dfs or 'fasttext' not in all_dfs: return

    p_w2v = all_dfs['w2v'].pivot(index="concept", columns="epoch", values="density")[EPOCHS]
    p_ft = all_dfs['fasttext'].pivot(index="concept", columns="epoch", values="density")[EPOCHS]

    d_w2v = p_w2v["eu"] - p_w2v["roman"]
    d_ft = p_ft["eu"] - p_ft["roman"]
    avg_delta = (d_w2v + d_ft) / 2

    sorter = avg_delta.sort_values(ascending=True).index
    p_w2v = p_w2v.reindex(sorter)
    p_ft = p_ft.reindex(sorter)

    vmin = min(p_w2v.min().min(), p_ft.min().min())
    vmax = max(p_w2v.max().max(), p_ft.max().max())

    # REDUCED SIZE: Was (14, 12) -> Now (7, 4)
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    cbar_ax = fig.add_axes([.92, .3, .02, .4])

    sns.heatmap(p_w2v, ax=axes[0], cmap="RdYlBu", vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", linewidths=.5, cbar=False,
                annot_kws={'size': 11})
    axes[0].set_title("Word2Vec", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis='x', labelsize=12, rotation=0)
    axes[0].tick_params(axis='y', labelsize=12, rotation=0)

    sns.heatmap(p_ft, ax=axes[1], cmap="RdYlBu", vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", linewidths=.5,
                cbar=True, cbar_ax=cbar_ax, cbar_kws={'label': 'Semantic Density'},
                annot_kws={'size': 11})
    axes[1].set_title("FastText", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='x', labelsize=12, rotation=0)

    cbar_ax.tick_params(labelsize=12)
    cbar_ax.set_ylabel('Semantic Density', size=12, weight='bold')

    #fig.suptitle("Semantic Erosion (Combined)", fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(out_dir / "heatmap_combined.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("   🖼️  Saved Combined Heatmap -> heatmap_combined.png")


def plot_slope_chart(df, m_type, out_dir):
    pivot = df.pivot(index="concept", columns="epoch", values="density")
    if "roman" not in pivot.columns or "eu" not in pivot.columns: return
    data = pivot[["roman", "eu"]].sort_values("roman", ascending=False)

    plt.figure(figsize=(9, 14))
    for concept, row in data.iterrows():
        color = "#d73027" if (row["eu"] - row["roman"]) < 0 else "#4575b4"
        alpha = 0.8 if abs(row["eu"] - row["roman"]) > 0.1 else 0.4
        width = 2 if abs(row["eu"] - row["roman"]) > 0.1 else 1
        plt.plot([0, 1], [row["roman"], row["eu"]], color=color, alpha=alpha, linewidth=width, marker='o')
        plt.text(-0.05, row["roman"], f"{concept}", ha='right', va='center', fontsize=11, color='#333')
        plt.text(1.05, row["eu"], f"{row['eu']:.2f}", ha='left', va='center', fontsize=11, color=color)

    plt.xticks([0, 1], ["Roman", "EU"], fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.ylabel("Semantic Density", fontsize=14)
    plt.title(f"Technicality Slope ({m_type.upper()})", fontsize=16, pad=20)
    sns.despine(left=True, bottom=True)
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"slope_chart_{m_type}.png", dpi=200)
    plt.close()
    print(f"   🖼️  Saved Slope Chart  -> slope_chart_{m_type}.png")


def plot_comparative_cloud(summary_data, out_dir):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    x = np.arange(len(EPOCHS))
    label_data = []

    for m_name, df in summary_data.items():
        pivot = df.pivot(index="concept", columns="epoch", values="density")[EPOCHS]
        for concept, row in pivot.iterrows():
            c = HIGHLIGHTS.get(concept, "#999999")
            ls = '-' if m_name == 'w2v' else '--'
            alpha = 0.9 if m_name == 'w2v' else 0.4
            width = 2.5 if m_name == 'w2v' else 1.5
            plt.plot(x, row.values, color=c, linewidth=width, alpha=alpha, linestyle=ls, zorder=3)
            if m_name == 'w2v':
                label_data.append({'y': row['eu'], 'text': concept.upper(), 'color': c})

        means = df.groupby('epoch')['density'].mean().reindex(EPOCHS)
        avg_color = '#1f77b4' if m_name == 'w2v' else '#ff7f0e'
        plt.plot(x, means.values, color=avg_color, linewidth=6, marker='o', markersize=9,
                 label=f"AVG {m_name.upper()}", zorder=5,
                 path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

    adjusted = adjust_labels_y(label_data, min_dist=(plt.ylim()[1] - plt.ylim()[0]) * 0.04)
    for lbl in adjusted:
        plt.text(2.05, lbl['y'], lbl['text'], color=lbl['color'], fontsize=11, fontweight='bold', va='center',
                 path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    plt.xticks(x, ["Roman", "Napoleon", "EU"], fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.ylabel("Semantic Density", fontsize=14, fontweight='bold')
    #plt.title("Evolution of Legal Concepts (Combined)", fontsize=18, pad=20)
    plt.legend(loc='lower left', fontsize=12, framealpha=0.9)
    plt.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.10)
    plt.savefig(out_dir / "comparative_model_trend_annotated.png", dpi=300)
    plt.close()
    print(f"   🖼️  Saved Cloud Chart  -> comparative_model_trend_annotated.png")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=======================================================")
    print("🚀 TECHNICALITY VISUALIZATION SUITE")
    print("=======================================================")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--fig_dir", required=True)
    parser.add_argument("--anchors", required=True)
    args = parser.parse_args()

    m_dir, met_dir, f_dir = Path(args.models_dir), Path(args.metrics_dir), Path(args.fig_dir)
    try:
        term_map = parse_anchors(args.anchors)
    except Exception as e:
        print(f"❌ Error: {e}"); return

    all_dfs = {}
    for m_type in MODELS:
        print(f"\n⚙️  PROCESSING MODEL: {m_type.upper()}")
        try:
            wvs = {ep: load_model(ep, m_type, m_dir) for ep in EPOCHS}
        except:
            print("   ⚠️  Model files missing."); continue

        results = []
        for concept, terms in term_map.items():
            for i, ep in enumerate(EPOCHS):
                d = get_density(wvs[ep], terms[i])
                if d: results.append({"concept": concept, "epoch": ep, "density": d})

        if results:
            df = pd.DataFrame(results)
            df.to_csv(met_dir / f"technicality_{m_type}.csv", index=False)
            all_dfs[m_type] = df
            plot_slope_chart(df, m_type, f_dir)
            # Individual heatmap with reduced size
            plot_heatmap(df, m_type, f_dir)

    if 'w2v' in all_dfs and 'fasttext' in all_dfs:
        print("\n📊 GENERATING COMBINED VISUALIZATIONS...")
        plot_comparative_cloud(all_dfs, f_dir)
        # Combined heatmap with reduced size
        plot_combined_heatmap(all_dfs, f_dir)

        # Stats
        def get_deltas(df):
            return df.pivot(index="concept", columns="epoch", values="density").eval("eu - roman")

        d1, d2 = get_deltas(all_dfs['w2v']), get_deltas(all_dfs['fasttext'])
        common = d1.index.intersection(d2.index)
        if len(common) > 0:
            r, p = pearsonr(d1[common], d2[common])
            with open(met_dir / "validation_stats.txt", "w") as f: f.write(f"PEARSON: r={r:.4f}, p={p:.4e}\n")
            print(f"   Validation Correlation: r={r:.4f}")

        # Stability
        means = all_dfs['w2v'].groupby('epoch')['density'].mean()
        ratio = abs(means['eu'] - means['napoleon']) / abs(means['napoleon'] - means['roman']) if abs(
            means['napoleon'] - means['roman']) != 0 else 0
        print(f"   Rupture Factor: {ratio:.1f}x")
        with open(met_dir / "validation_stats.txt", "a") as f:
            f.write(f"Rupture Factor: {ratio:.1f}x\n")


if __name__ == "__main__":
    main()