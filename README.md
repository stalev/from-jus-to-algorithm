# From Jus to Algorithm: A Cross-Epoch Analysis of Semantic Density

![Status](https://img.shields.io/badge/Status-In--Progress-orange) 
![License](https://img.shields.io/badge/License-MIT-green)

> [!NOTE]
> **Preprint status:** Under review. Methodology subject to refinement. 
> For citation purposes, please refer to the `CITATION.cff` file.

---

This repository contains the data and code for a diachronic study of legal language evolution. We quantify the semantic transformation of foundational legal concepts over 1,500 years—from Roman law and the Napoleonic Code to modern EU digital regulations.

**Key finding:** Our geometric analysis reveals a systematic reduction in semantic density, providing empirical evidence for the conceptual "dilution" of legal terminology in the digital age.

### Methodology Overview
1. **Corpus Alignment:** Language-specific preprocessing and lemmatization of Latin, French, and English legal texts.
2. **Vector Space Training:** Generation of 300-dimensional embeddings using Word2Vec (SGNS) and FastText.
3. **Procrustes Alignment:** Transformation of epoch-specific spaces into a unified coordinate system using anchor terms.
4. **Density Calculation:** Measuring the "tightness" of semantic neighborhoods ($k=25$) as a proxy for conceptual stability.

## Requirements
* **Python 3.11** is strictly required for compatibility with the provided `requirements.txt`.
* At least **16GB of RAM** is recommended for embedding training.

## Quick Start

Follow these steps to set up the project and run the research pipeline locally.

---

### 0. Clone the Repository
```bash
git clone https://github.com/stalev/from-jus-to-algorithm.git
```

### 1. Repository Structure

Before running the Snakemake pipeline, the repository has the following clean structure. Note that the output directories (such as `results/` and `logs/`) are generated automatically during the pipeline execution.

```text
.
├── LICENSE
├── README.md
├── Snakefile              # The main Snakemake pipeline definition
├── requirements.txt       # Python dependencies required for the project
├── config/
│   └── config.yaml        # Central configuration (epochs, models, input/output dirs)
├── data/
│   ├── raw/               # Raw JSON corpora (included in repo)
│   │   ├── eu.json        # Code Justinian, books I-XII 
│   │   ├── napoleon.json  # Napoleonic codes
│   │   └── roman.json     # EU Digital Acquis, 10 norms
│   └── reference/         # Manually curated dictionaries for analysis
│       ├── alignment_anchors.json # Dictionary for orthogonal Procrustes alignment
│       └── legal_anchors.json     # Dictionary of target legal concepts for displacement analysis
├── gadgets/               # Standalone helper scripts (not part of the main pipeline)
│   ├── check_anchors.py
│   ├── compare_alignment.py
│   └── json_stats.py
└── scripts/               # Core Python scripts executed by Snakemake
    ├── preprocess_*.py    # Language-specific tokenization and lemmatization (LA, FR, EN)
    ├── extract_lemmas.py  # POS-filtering (keeps only Nouns, Verbs, Adjectives)
    ├── analyze_vocab.py   # Vocabulary statistics generation
    ├── train_models.py    # Generates epoch-specific Word2Vec and FastText embeddings
    ├── align_models.py    # Orthogonal Procrustes alignment implementation
    ├── assess_*.py        # Post-alignment quality and anchor stability checks
    ├── plot_*.py          # Visualization scripts for corpus distribution
    ├── technicality_analysis.py          # Semantic Density calculation
    └── semantic_displacement_analysis.py # Concept mobility and kinetic visualization
```

### 2. Set up the local environment
```bash
# Switch to the project folder 
cd from-jus-to-algorithm

# Create a virtual environment
python3 -m venv .venv
 
# Activate the environment 
source .venv/bin/activate # for bash/zsh
source .venv/bin/activate.fish # For fish

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the research pipeline
Complete pipeline run might take up to 30 min (recommended: at least 16GB RAM)

```bash
snakemake --cores 4
```

### 4. Force Re-run 
If you need to re-run the entire pipeline from scratch (ignoring existing output), use the force flag:

```bash
snakemake --cores 4 --forceall
```

## Citation

If you use this methodology, code, or the provided embeddings in your research, please cite this work as follows:

> Refer to the [`CITATION.cff`](./CITATION.cff) file for the metadata and preferred citation format: Levchenko, S., Kovalyov, O., & Eteris, E. (2026). From Jus to Algorithm: A Cross-Epoch Analysis of Semantic Density in Legal Corpora. GitHub Repository: https://github.com/stalev/from-jus-to-algorithm
