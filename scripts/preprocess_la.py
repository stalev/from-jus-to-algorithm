#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
import logging
import sys
from pathlib import Path
import stanza

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def init_pipeline():
    """
    Initializes Stanza for Latin.
    use_gpu=False prevents crashes on Apple Silicon (M1/M2/M3).
    Processors:
      - tokenize: Split text into words
      - pos: Identify parts of speech (Noun, Verb, etc.) to filter noise
      - lemma: Convert words to base form
    """
    logger.info("Checking Latin models...")
    stanza.download('la', processors='tokenize,lemma,pos', verbose=False)
    logger.info("Initializing Stanza pipeline...")
    return stanza.Pipeline('la', processors='tokenize,lemma,pos', use_gpu=False, logging_level='WARN')


def process_paragraph(text, nlp):
    """
    Processing logic for a single text block.
    Returns: (lemmatized_string, list_of_excluded_lemmas)
    """
    # 1. Clean Roman Law markup (*, <, >)
    for char in ['*', '<', '>']:
        if char in text:
            text = text.split(char)[0]

    if not text.strip():
        return "", []

    try:
        doc = nlp(text)
    except Exception as e:
        logger.warning(f"Stanza failed: {e}")
        return "", []

    lemmas = []
    excluded = set()

    # UPOS tags to EXCLUDE (Stopwords logic based on Universal Dependencies)
    # PUNCT=Punctuation, SYM=Symbols, NUM=Numbers
    # ADP=Adpositions (prepositions), DET=Determiners, CCONJ/SCONJ=Conjunctions
    # PRON=Pronouns, AUX=Auxiliary verbs (esse), PART=Particles, INTJ=Interjections, X=Unknown
    EXCLUDED_POS = {
        'PUNCT', 'SYM', 'NUM', 'X',  'ADV',
        'ADP', 'DET', 'CCONJ', 'SCONJ', 'PRON', 'AUX', 'PART', 'INTJ'
    }

    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower() if word.lemma else word.text.lower()

            # Filter by Part of Speech
            if word.pos in EXCLUDED_POS:
                excluded.add(lemma)
                continue

            # Filter single characters (often noise in OCR/old texts)
            if len(lemma) < 2:
                excluded.add(lemma)
                continue

            lemmas.append(lemma)

    return " ".join(lemmas), sorted(list(excluded))


def process_recursive(data, nlp):
    """
    Recursively traverses the JSON structure to find 'paragraphs'.
    Modifies the dictionary in-place.
    """
    # If we found a list of paragraphs (the leaf node structure in Roman corpus)
    if isinstance(data, list) and len(data) > 0 and "fragment_id" in data[0]:
        for p in data:
            raw_text = p.get("original_text", "")
            lemmas, excluded = process_paragraph(raw_text, nlp)
            p["lemmatized_text"] = lemmas
            p["excluded_lemmas"] = excluded
        return

    # If it's a dictionary (Part, Book, Title), go deeper
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                process_recursive(value, nlp)

    # If it's a list of objects (e.g. list of books), iterate
    elif isinstance(data, list):
        for item in data:
            process_recursive(item, nlp)


def preprocess_file(file_path, out_dir, nlp):
    path = Path(file_path)

    # 1. Load JSON
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {path}")
            sys.exit(1)

    logger.info(f"Processing structure for {path.name}...")

    # 2. Process In-Place (preserving structure)
    # We pass the whole structure. The function searches for paragraphs deep inside.
    process_recursive(data, nlp)

    # 3. Save Normalized JSON (Identical structure to input + new fields)
    out_path = Path(out_dir) / f"{path.stem}_norm.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved normalized Latin corpus to: {out_path}")


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
        logger.exception("Critical failure during processing")
        sys.exit(1)