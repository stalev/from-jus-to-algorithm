import argparse
import json
import numpy as np
import scipy.linalg
from pathlib import Path
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity


def load_model(path):
    """Load W2V or FastText model with debug print."""
    path_str = str(path)
    print(f"    [DEBUG] Loading: {Path(path_str).name}...", end=" ", flush=True)
    try:
        if "fasttext" in path_str:
            m = FastText.load(path_str)
        else:
            m = Word2Vec.load(path_str)
        print("OK.")
        return m
    except Exception as e:
        print(f"FAILED! Error: {e}")
        return None


def get_common_vectors(model_a, model_b, anchors, lang_a, lang_b, debug_tag=""):
    """
    Extracts vectors for anchors that exist in both models.
    """
    vecs_a = []
    vecs_b = []
    valid_anchors = []
    missing_a = []
    missing_b = []

    for item in anchors:
        w_a = item.get(lang_a)
        w_b = item.get(lang_b)

        in_a = w_a in model_a.wv
        in_b = w_b in model_b.wv

        if in_a and in_b:
            vecs_a.append(model_a.wv[w_a])
            vecs_b.append(model_b.wv[w_b])
            valid_anchors.append(item)
        else:
            if not in_a: missing_a.append(w_a)
            if not in_b: missing_b.append(w_b)

    # Debug info
    count = len(valid_anchors)
    total = len(anchors)
    print(f"    [DEBUG] {debug_tag}: Found {count}/{total} common anchors.")

    if count == 0:
        print(f"    [DEBUG] WARNING: No intersections found!")
        if len(missing_a) > 0: print(f"    [DEBUG] Missing in A (first 3): {missing_a[:3]}")
        if len(missing_b) > 0: print(f"    [DEBUG] Missing in B (first 3): {missing_b[:3]}")

    return np.array(vecs_a), np.array(vecs_b), count


def compute_metrics(src_raw, tgt_raw, src_aligned, tgt_aligned, anchors, src_lang, tgt_lang):
    """
    Computes alignment metrics.
    """

    # --- 1. BEFORE: Raw vs Raw ---
    A_raw, B_raw, count_raw = get_common_vectors(src_raw, tgt_raw, anchors, src_lang, tgt_lang, debug_tag="Raw check")

    if count_raw == 0:
        return None

    # Normalize
    A_raw_norm = A_raw / np.linalg.norm(A_raw, axis=1, keepdims=True)
    B_raw_norm = B_raw / np.linalg.norm(B_raw, axis=1, keepdims=True)
    cos_before = np.sum(A_raw_norm * B_raw_norm, axis=1).mean()

    # --- 2. AFTER: Aligned vs Aligned ---
    A_aln, B_aln, count_aln = get_common_vectors(src_aligned, tgt_aligned, anchors, src_lang, tgt_lang,
                                                 debug_tag="Aligned check")

    if count_aln == 0:
        return None

    # --- DEBUG CHECK: Did alignment move ANY vectors? ---
    # Проверяем, сдвинулся ли Source
    src_moved = not np.allclose(A_raw, A_aln)
    # Проверяем, сдвинулся ли Target (сравниваем Raw Target и Aligned Target)
    # Важно: B_raw и B_aln - это векторы Target (Napoleon/EU)
    tgt_moved = not np.allclose(B_raw, B_aln)

    if not src_moved and not tgt_moved:
        print("    [DEBUG] ⚠️ CRITICAL WARNING: NEITHER Source NOR Target vectors moved.")
        print("    [DEBUG] ⚠️ This implies NO alignment happened.")
    elif not src_moved and tgt_moved:
        print("    [DEBUG] Info: Source (Roman) stayed fixed, Target (Napoleon/EU) rotated to match it.")
    elif src_moved and not tgt_moved:
        print("    [DEBUG] Info: Target stayed fixed, Source rotated to match it (Standard Procrustes).")
    else:
        print("    [DEBUG] Info: Both models moved (Common Space Alignment).")

    # Normalize
    A_aln_norm = A_aln / np.linalg.norm(A_aln, axis=1, keepdims=True)
    B_aln_norm = B_aln / np.linalg.norm(B_aln, axis=1, keepdims=True)
    cos_after = np.sum(A_aln_norm * B_aln_norm, axis=1).mean()

    # --- Metrics ---
    gain = cos_after - cos_before

    frob_error = np.linalg.norm(A_aln - B_aln, ord='fro')

    # Orthogonality Error
    R, _ = scipy.linalg.orthogonal_procrustes(A_raw, B_raw)
    I = np.eye(R.shape[0])
    ortho_error = np.linalg.norm(R.T @ R - I, ord='fro')

    return {
        "before": cos_before,
        "after": cos_after,
        "gain": gain,
        "frobenius": frob_error,
        "orthogonality": ortho_error,
        "count": count_aln
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Procrustes Alignment Quality")
    parser.add_argument("--anchors", required=True, help="Path to alignment_anchors.json")
    parser.add_argument("--models_dir", required=True, help="Path to results/models directory")
    args = parser.parse_args()

    # Load Anchors
    print(f"\n[STEP 1] Loading Anchors from: {args.anchors}")
    with open(args.anchors, 'r') as f:
        content = "".join([line for line in f if not line.strip().startswith("//")])
        anchors = json.loads(content)

    print(f"    Loaded {len(anchors)} triplets.")
    print(f"    [DEBUG] Sample anchors (first 2): {anchors[:2]}")

    mdir = Path(args.models_dir)

    tasks = [
        ("roman", "napoleon", "la", "fr"),
        ("roman", "eu", "la", "en")
    ]

    model_types = ["w2v", "fasttext"]
    results = {}

    print("\n[STEP 2] Processing Models...")

    for src_name, tgt_name, sl, tl in tasks:
        results[tgt_name] = {}

        for mtype in model_types:
            print(f"\n=== COMPARISON: {src_name.upper()} -> {tgt_name.upper()} ({mtype.upper()}) ===")

            # Paths
            path_src_raw = mdir / f"{src_name}_{mtype}.model"
            path_tgt_raw = mdir / f"{tgt_name}_{mtype}.model"
            path_src_aln = mdir / f"{src_name}_aligned_{mtype}.model"
            path_tgt_aln = mdir / f"{tgt_name}_aligned_{mtype}.model"

            # Check files
            if not (
                    path_src_raw.exists() and path_tgt_raw.exists() and path_src_aln.exists() and path_tgt_aln.exists()):
                print(f"    [ERROR] Missing files for {mtype}. Skipping.")
                results[tgt_name][mtype] = None
                continue

            # Load all 4
            m_src_raw = load_model(path_src_raw)
            m_tgt_raw = load_model(path_tgt_raw)
            m_src_aln = load_model(path_src_aln)
            m_tgt_aln = load_model(path_tgt_aln)

            if not all([m_src_raw, m_tgt_raw, m_src_aln, m_tgt_aln]):
                print("    [ERROR] Failed to load one or more models.")
                continue

            # Compute
            print("    [DEBUG] Computing metrics...")
            metrics = compute_metrics(
                m_src_raw, m_tgt_raw,
                m_src_aln, m_tgt_aln,
                anchors, sl, tl
            )

            results[tgt_name][mtype] = metrics

            if metrics:
                print(f"    [RESULT] Cosine Gain: {metrics['gain']:.4f}")

    # --- Print Table ---
    print("\n" + "=" * 80)
    print("Table 2: Quality of Procrustes Alignment Across Epochs")
    print("-" * 80)
    print(f"{'Metric':<25} {'Roman -> Napoleon':<25} {'Roman -> EU':<25}")
    print(f"{'':<25} {'W2V':<10} {'FT':<10}   {'W2V':<10} {'FT':<10}")
    print("-" * 80)

    def get_val(tgt, mtype, key, fmt="{:.3f}"):
        res = results.get(tgt, {}).get(mtype)
        if res:
            return fmt.format(res[key])
        return "-"

    rows = [
        ("Anchor cosine (before)", "before", "{:.3f}"),
        ("Anchor cosine (after)", "after", "{:.3f}"),
        ("Cosine gain (d)", "gain", "+{:.3f}"),
        ("Frobenius error", "frobenius", "{:.2f}"),
        ("Orthogonality error", "orthogonality", "{:.2e}"),
        ("Anchor count", "count", "{}")
    ]

    for label, key, fmt in rows:
        nap_w2v = get_val("napoleon", "w2v", key, fmt)
        nap_ft = get_val("napoleon", "fasttext", key, fmt)
        eu_w2v = get_val("eu", "w2v", key, fmt)
        eu_ft = get_val("eu", "fasttext", key, fmt)

        print(f"{label:<25} {nap_w2v:<10} {nap_ft:<10}   {eu_w2v:<10} {eu_ft:<10}")

    print("-" * 80)


if __name__ == "__main__":
    main()