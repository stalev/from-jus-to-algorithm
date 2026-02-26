import argparse
import json
import numpy as np
from gensim.models import Word2Vec, FastText
from scipy.linalg import orthogonal_procrustes
import os


def align(source_path, target_path, anchors_path, output_path, source_lang, target_lang):
    """
    Performs Orthogonal Procrustes Alignment to rotate the source model
    into the coordinate system of the target model using anchor points.
    """
    # Determine model type based on the file name
    is_fasttext = 'fasttext' in source_path.lower()
    load_func = FastText.load if is_fasttext else Word2Vec.load

    print(f"Loading models: {source_path} and {target_path}...")
    src_model = load_func(source_path)
    tgt_model = load_func(target_path)

    # Load semantic anchors (supports skipping lines starting with //)
    with open(anchors_path, 'r', encoding='utf-8') as f:
        content = "".join([line for line in f if not line.strip().startswith("//")])
        anchors = json.loads(content)

    # Build vector matrices for common anchors
    src_vecs = []
    tgt_vecs = []

    for a in anchors:
        s_word = a.get(source_lang)
        t_word = a.get(target_lang)

        # Ensure words exist in their respective model vocabularies
        if s_word and t_word and s_word in src_model.wv and t_word in tgt_model.wv:
            src_vecs.append(src_model.wv[s_word])
            tgt_vecs.append(tgt_model.wv[t_word])

    if len(src_vecs) < 2:
        raise ValueError(f"Error: Only {len(src_vecs)} anchors found. Need at least 2 for Procrustes rotation.")

    A = np.array(src_vecs)
    B = np.array(tgt_vecs)

    # Compute the Orthogonal Procrustes matrix (R)
    # This finds the rotation matrix R that minimizes ||A @ R - B||^2
    R, _ = orthogonal_procrustes(A, B)

    # Apply transformation to the main vocabulary vectors
    src_model.wv.vectors = src_model.wv.vectors @ R

    # For FastText, it is crucial to transform n-gram vectors to maintain OOV support
    if is_fasttext:
        if hasattr(src_model.wv, 'vectors_ngrams'):
            src_model.wv.vectors_ngrams = src_model.wv.vectors_ngrams @ R
        if hasattr(src_model.wv, 'vectors_vocab'):
            src_model.wv.vectors_vocab = src_model.wv.vectors_vocab @ R

    # Save the aligned model
    src_model.save(output_path)
    print(f"Successfully aligned {source_lang} to {target_lang} using {len(src_vecs)} anchors.")
    print(f"Aligned model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align word embeddings using Orthogonal Procrustes.")
    parser.add_argument("--source", required=True, help="Path to the source model to be rotated")
    parser.add_argument("--target", required=True, help="Path to the reference (target) model")
    parser.add_argument("--anchors", required=True, help="Path to alignment_anchors.json")
    parser.add_argument("--output", required=True, help="Output path for the aligned model")
    parser.add_argument("--source_lang", required=True, help="Source language key in JSON (e.g., fr, en)")
    parser.add_argument("--target_lang", required=True, help="Target language key in JSON (e.g., la)")

    args = parser.parse_args()

    align(args.source, args.target, args.anchors, args.output, args.source_lang, args.target_lang)