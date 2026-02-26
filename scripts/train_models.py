import json
import logging
import argparse
import multiprocessing
from pathlib import Path
from gensim.models import Word2Vec, FastText

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_sentences_from_json(file_path):
    """Recursively extracts lemmatized sentences from the JSON structure."""
    if not Path(file_path).exists():
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []

    sentences = []

    def walk(obj):
        if isinstance(obj, dict):
            if "lemmatized_text" in obj and obj["lemmatized_text"]:
                sentences.append(obj["lemmatized_text"].split())
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for i in obj: walk(i)

    walk(data)
    return sentences


def train_model(input_json, output_prefix, model_type="w2v"):
    """Trains a specific model type for a given input file."""
    sentences = get_sentences_from_json(input_json)
    if not sentences:
        logger.error(f"No sentences loaded from {input_json}")
        return

    cores = multiprocessing.cpu_count()

    if model_type == "w2v":
        logger.info(f"Training Word2Vec for {output_prefix}...")
        model = Word2Vec(sentences=sentences, vector_size=300, window=5,
                         min_count=5, sg=1, workers=cores, epochs=15)
        model.save(f"{output_prefix}_w2v.model")
    else:
        logger.info(f"Training FastText for {output_prefix}...")
        model = FastText(sentences=sentences, vector_size=300, window=5,
                         min_count=5, sg=1, min_n=3, max_n=6, workers=cores, epochs=15)
        model.save(f"{output_prefix}_fasttext.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Output path/prefix")
    parser.add_argument("--type", type=str, choices=["w2v", "fasttext"], default="w2v")
    args = parser.parse_args()

    train_model(args.input, args.output_prefix, args.type)