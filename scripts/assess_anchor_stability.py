import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


def load_vocab_counts(vocab_dir):
    """
    Loads word frequency counts from CSV vocabulary files.
    """
    counts = {'la': {}, 'fr': {}, 'en': {}}
    files = {'la': 'roman.csv', 'fr': 'napoleon.csv', 'en': 'eu.csv'}

    print(f"Loading vocabulary stats from {vocab_dir}...")
    for lang, filename in files.items():
        path = Path(vocab_dir) / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                term_col = df.columns[0]
                # Assume second column is count; if missing, use first
                count_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                vocab_dict = pd.Series(df[count_col].values, index=df[term_col]).to_dict()
                counts[lang] = vocab_dict
            except Exception as e:
                print(f"  [{lang}] Warning: Error reading {filename}: {e}")
        else:
            print(f"  [{lang}] Warning: File {filename} not found.")
    return counts


def get_valid_anchors(anchors, models):
    """
    Filters anchors to keep only those present in the intersection
    of all provided models (La, Fr, En).
    """
    valid = []
    for a in anchors:
        if (a['la'] in models['la'].wv and
                a['fr'] in models['fr'].wv and
                a['en'] in models['en'].wv):
            valid.append(a)
    return valid


def get_so_vector(model, target_word, basis_anchors, lang_key):
    """
    Constructs a Second-Order (SO) vector for a target word.
    """
    target_vec = model.wv[target_word].reshape(1, -1)
    basis_vectors = np.array([model.wv[a[lang_key]] for a in basis_anchors])
    return sk_cosine_similarity(target_vec, basis_vectors).flatten()


def get_stability_label(avg_score):
    """
    Assigns a stability category based on the average cosine similarity score.
    Clean text version (no LaTeX).
    """
    if avg_score > 0.85: return "ROBUST"
    if avg_score > 0.70: return "Stable"
    if avg_score > 0.50: return "Drift"
    return "Noisy"


def calculate_metrics(models, basis, item):
    """
    Calculates detailed stability metrics for a specific triplet.
    """
    if not (item['la'] in models['la'].wv and
            item['fr'] in models['fr'].wv and
            item['en'] in models['en'].wv):
        return None

    try:
        so_la = get_so_vector(models['la'], item['la'], basis, 'la')
        so_fr = get_so_vector(models['fr'], item['fr'], basis, 'fr')
        so_en = get_so_vector(models['en'], item['en'], basis, 'en')

        s_la_fr = sk_cosine_similarity(so_la.reshape(1, -1), so_fr.reshape(1, -1))[0][0]
        s_fr_en = sk_cosine_similarity(so_fr.reshape(1, -1), so_en.reshape(1, -1))[0][0]

        avg = (s_la_fr + s_fr_en) / 2
        label = get_stability_label(avg)

        return {"la_fr": s_la_fr, "fr_en": s_fr_en, "avg": avg, "label": label}
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Assess semantic stability of anchors across 3 periods.")
    parser.add_argument("--anchors", required=True, help="Path to JSON with anchors")
    parser.add_argument("--models_dir", required=True, help="Directory containing .model files")
    parser.add_argument("--vocab_dir", required=True, help="Directory containing .csv vocab files")
    args = parser.parse_args()

    # 1. Load Anchors
    with open(args.anchors, 'r', encoding='utf-8') as f:
        content = "".join([line for line in f if not line.strip().startswith("//")])
        anchors = json.loads(content)

    # Sort anchors alphabetically by the English term
    anchors.sort(key=lambda x: x['en'])

    # 2. Load Vocabulary Counts
    vocab_counts = load_vocab_counts(args.vocab_dir)

    # 3. Load Models
    print("Loading models...")
    try:
        w2v_models = {
            'la': Word2Vec.load(str(Path(args.models_dir) / "roman_w2v.model")),
            'fr': Word2Vec.load(str(Path(args.models_dir) / "napoleon_w2v.model")),
            'en': Word2Vec.load(str(Path(args.models_dir) / "eu_w2v.model"))
        }
        ft_models = {
            'la': FastText.load(str(Path(args.models_dir) / "roman_fasttext.model")),
            'fr': FastText.load(str(Path(args.models_dir) / "napoleon_fasttext.model")),
            'en': FastText.load(str(Path(args.models_dir) / "eu_fasttext.model"))
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. Establish Validation Basis
    w2v_basis = get_valid_anchors(anchors, w2v_models)
    ft_basis = get_valid_anchors(anchors, ft_models)

    # 5. Generate Output Table (Text Format)

    # Define header layout
    # Columns: Triplet (55 chars) | W2V Stats (26 chars) | W2V Status | FT Stats (26 chars) | FT Status
    header_fmt = "{:<55} | {:<26} {:<9} || {:<26} {:<9}"

    print("\n" + "=" * 138)
    print(header_fmt.format(
        "TRIPLET [LA(n) / FR(n) / EN(n)]",
        "W2V [LA-FR  FR-EN   AVG ]", "STATUS",
        "FT  [LA-FR  FR-EN   AVG ]", "STATUS"
    ))
    print("=" * 138)

    for item in anchors:
        la_word = item['la']
        fr_word = item['fr']
        en_word = item['en']

        # Get frequencies
        c_la = vocab_counts['la'].get(la_word, "?")
        c_fr = vocab_counts['fr'].get(fr_word, "?")
        c_en = vocab_counts['en'].get(en_word, "?")

        # Format label string
        label_str = f"{la_word}({c_la}) / {fr_word}({c_fr}) / {en_word}({c_en})"
        # Truncate label if too long to maintain table alignment
        if len(label_str) > 53:
            label_str = label_str[:50] + "..."

        # Calculate W2V Metrics
        w_res = calculate_metrics(w2v_models, w2v_basis, item)
        if w_res:
            w_str = f"{w_res['la_fr']:>6.2f}  {w_res['fr_en']:>6.2f}  {w_res['avg']:>6.2f}"
            w_stat = w_res['label']
        else:
            w_str = "   -       -       -  "
            w_stat = "Missing"

        # Calculate FastText Metrics
        f_res = calculate_metrics(ft_models, ft_basis, item)
        if f_res:
            f_str = f"{f_res['la_fr']:>6.2f}  {f_res['fr_en']:>6.2f}  {f_res['avg']:>6.2f}"
            f_stat = f_res['label']
        else:
            f_str = "   -       -       -  "
            f_stat = "Missing"

        # Print formatted row
        print(f"{label_str:<55} | {w_str} {w_stat:<9} || {f_str} {f_stat:<9}")

    print("-" * 138)


if __name__ == "__main__":
    main()