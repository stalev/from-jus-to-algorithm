import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_combined_distribution(file_map, output_file):
    plt.figure(figsize=(10, 6))

    # Set plot style
    sns.set_style("whitegrid")
    # Define specific colors for known corpora
    colors = {'roman': '#e74c3c', 'napoleon': '#3498db', 'eu': '#2ecc71'}

    for name, csv_path in file_map.items():
        path = Path(csv_path)

        try:
            # Read CSV. Assume format: word, frequency
            df = pd.read_csv(path)
            # If columns are unnamed or missing, assume the 2nd column is frequency; otherwise use the 1st
            count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            # Sort by frequency descending
            counts = df[count_col].sort_values(ascending=False).reset_index(drop=True)

            # Plot Log-Log graph (Rank vs Frequency)
            plt.loglog(counts.index + 1, counts.values,
                       label=name.capitalize(),
                       color=colors.get(name, 'gray'),
                       linewidth=2, alpha=0.8)

        except Exception as e:
            print(f"Warning: Could not process {name} ({csv_path}): {e}")

    plt.title("Word Frequency Distribution (Zipf's Law)", fontsize=14)
    plt.xlabel("Rank (log scale)", fontsize=12)
    plt.ylabel("Frequency (log scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Combined plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Updated to match the flags used in your Snakefile error log
    parser.add_argument("--roman", required=True, help="Path to Roman CSV")
    parser.add_argument("--napoleon", required=True, help="Path to Napoleon CSV")
    parser.add_argument("--eu", required=True, help="Path to EU CSV")

    parser.add_argument("--output", required=True, help="Output PNG file")

    args = parser.parse_args()

    # Create a dictionary mapping names to file paths
    input_map = {
        'roman': args.roman,
        'napoleon': args.napoleon,
        'eu': args.eu
    }

    plot_combined_distribution(input_map, args.output)