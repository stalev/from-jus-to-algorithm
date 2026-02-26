import json
import argparse
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec, FastText
from gensim import matutils


def get_sim(model1, word1, model2, word2):
    """Calculates cosine similarity between two models."""
    if word1 not in model1.wv or word2 not in model2.wv:
        return None
    v1 = matutils.unitvec(model1.wv[word1])
    v2 = matutils.unitvec(model2.wv[word2])
    return float(np.dot(v1, v2))


def main():
    parser = argparse.ArgumentParser(description="Unified Alignment Evaluation Report")
    parser.add_argument("--anchors", default="data/reference/alignment_anchors.json")
    parser.add_argument("--models_dir", default="results/models")
    parser.add_argument("--mtype", default="w2v", choices=["w2v", "fasttext"])
    args = parser.parse_args()

    # 1. Load Anchors
    with open(args.anchors, 'r', encoding='utf-8') as f:
        content = "".join([line for line in f if not line.strip().startswith("//")])
        anchors = json.loads(content)

    m_dir = Path(args.models_dir)
    load_func = FastText.load if args.mtype == "fasttext" else Word2Vec.load

    # 2. Load all 6 models (3 original, 3 aligned)
    print(f"Loading {args.mtype.upper()} models into memory...")
    try:
        orig = {
            'la': load_func(str(m_dir / f"roman_{args.mtype}.model")),
            'fr': load_func(str(m_dir / f"napoleon_{args.mtype}.model")),
            'en': load_func(str(m_dir / f"eu_{args.mtype}.model"))
        }
        algn = {
            'la': load_func(str(m_dir / f"roman_aligned_{args.mtype}.model")),
            'fr': load_func(str(m_dir / f"napoleon_aligned_{args.mtype}.model")),
            'en': load_func(str(m_dir / f"eu_aligned_{args.mtype}.model"))
        }
    except Exception as e:
        print(f"Critical error loading models: {e}")
        return

    # 3. Header
    # B: Before, A: After, G: Gain
    print(f"\n{'TRIPLET (LA/FR/EN)':<40} | {'LA-FR (B/A)':<15} | {'LA-EN (B/A)':<15} | {'GAINS (FR/EN)'}")
    print("-" * 105)

    gains_fr, gains_en = [], []

    for a in anchors:
        la, fr, en = a['la'], a['fr'], a['en']

        # LA-FR Metrics
        s_fr_b = get_sim(orig['la'], la, orig['fr'], fr)
        s_fr_a = get_sim(algn['la'], la, algn['fr'], fr)

        # LA-EN Metrics
        s_en_b = get_sim(orig['la'], la, orig['en'], en)
        s_en_a = get_sim(algn['la'], la, algn['en'], en)

        # Format strings for cells
        fr_cell = f"{s_fr_b:>6.2f}→{s_fr_a:<6.2f}" if s_fr_b is not None else "   MISSING    "
        en_cell = f"{s_en_b:>6.2f}→{s_en_a:<6.2f}" if s_en_b is not None else "   MISSING    "

        gain_str = ""
        if s_fr_b is not None and s_fr_a is not None:
            g_fr = s_fr_a - s_fr_b
            gains_fr.append(g_fr)
            gain_str += f"{g_fr:>+6.2f}"
        else:
            gain_str += "  -   "

        gain_str += " / "

        if s_en_b is not None and s_en_a is not None:
            g_en = s_en_a - s_en_b
            gains_en.append(g_en)
            gain_str += f"{g_en:>+6.2f}"
        else:
            gain_str += "  -   "

        triplet_label = f"{la}/{fr}/{en}"
        print(f"{triplet_label[:40]:<40} | {fr_cell:<15} | {en_cell:<15} | {gain_str}")

    # 4. Final Summary
    print("-" * 105)
    if gains_fr and gains_en:
        avg_fr = np.mean(gains_fr)
        avg_en = np.mean(gains_en)
        print(
            f"{'AVERAGE SYSTEM GAIN':<40} | LA-FR: {avg_fr:>+7.4f} | LA-EN: {avg_en:>+7.4f} | TOTAL: {np.mean(gains_fr + gains_en):>+7.4f}")


if __name__ == "__main__":
    main()