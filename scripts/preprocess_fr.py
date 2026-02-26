#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
import logging
import sys
from pathlib import Path
import stanza

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def init_pipeline():
    """
    Initializes Stanza for French.
    Consistent with Latin pipeline methodology.
    """
    logger.info("Checking French models...")
    stanza.download('fr', processors='tokenize,lemma,pos', verbose=False)
    logger.info("Initializing Stanza pipeline (fr)...")
    # use_gpu=False for stability on Mac
    return stanza.Pipeline('fr', processors='tokenize,lemma,pos', use_gpu=False, logging_level='WARN')


def process_paragraph(text, nlp):
    """
    Unified processing logic (Same as Latin).
    """
    if not text or not text.strip():
        return "", []

    try:
        doc = nlp(text)
    except Exception as e:
        logger.warning(f"Stanza failed: {e}")
        return "", []

    lemmas = []
    excluded = set()

    # Universal Dependencies tags to EXCLUDE (Identical to Latin script)
    EXCLUDED_POS = {
        'PUNCT', 'SYM', 'NUM', 'X',  'ADV',
        'ADP', 'DET', 'CCONJ', 'SCONJ', 'PRON', 'AUX', 'PART', 'INTJ'
    }

    for sent in doc.sentences:
        for word in sent.words:
            # Stanza lemmas are usually lowercase, but force it just in case
            lemma = word.lemma.lower() if word.lemma else word.text.lower()

            # 1. Filter by Part of Speech
            if word.pos in EXCLUDED_POS:
                excluded.add(lemma)
                continue

            # 2. Filter short noise (optional, consistent with Latin)
            if len(lemma) < 2:
                excluded.add(lemma)
                continue

            lemmas.append(lemma)

    return " ".join(lemmas), sorted(list(excluded))


def process_recursive(data, nlp):
    """
    Traverses JSON to find objects with 'original_text' or 'text'.
    Works for both flat lists and hierarchical trees.
    """
    if isinstance(data, list):
        for item in data:
            process_recursive(item, nlp)

    elif isinstance(data, dict):
        # Check if this object is a paragraph
        # We check both keys just in case the input format varies
        raw_text = data.get("original_text") or data.get("text")

        if raw_text and isinstance(raw_text, str):
            lemmas, excluded = process_paragraph(raw_text, nlp)
            data["lemmatized_text"] = lemmas
            data["excluded_lemmas"] = excluded

        # Continue recursion into dictionary values
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                process_recursive(value, nlp)


def preprocess_file(file_path, out_dir, nlp):
    path = Path(file_path)

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {path}")
            sys.exit(1)

    logger.info(f"Processing {path.name} with Stanza (French)...")
    process_recursive(data, nlp)

    out_path = Path(out_dir) / f"{path.stem}_norm.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    try:
        nlp_pipeline = init_pipeline()
        preprocess_file(args.file, args.out_dir, nlp_pipeline)
    except Exception as e:
        logger.exception("Critical failure")
        sys.exit(1)