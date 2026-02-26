#!/usr/bin/env python3
# -----------------------------------------------------------
# 📊 JSON Corpus Statistics Utility
#
# Counts:
#   • words  → ONLY from `original_text`
#   • lemmas → ONLY from `lemmatized_text` (if present)
#
# Usage:
#   python gadgets/json_stats.py data/normalized/
#   python gadgets/json_stats.py data/normalized --show_parts
#   python gadgets/json_stats.py data/normalized --tree
# -----------------------------------------------------------

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+", re.UNICODE)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
# def count_words(text: str) -> int:
#    return len(WORD_RE.findall(text)) if text else 0
def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def count_lemmas(text: str) -> int:
    return len(text.split()) if text else 0


def snippet(text: str, length: int = 100) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text[:length] + ("…" if len(text) > length else "")


def resolve_input_paths(paths: List[str]) -> List[Path]:
    resolved: List[Path] = []

    for p in paths:
        for token in p.split(","):
            token = token.strip()
            if not token:
                continue

            path = Path(token)

            if any(w in token for w in ["*", "?", "["]):
                resolved.extend(
                    f for f in Path().glob(token)
                    if f.is_file() and f.suffix == ".json"
                )
            elif path.is_file() and path.suffix == ".json":
                resolved.append(path)
            elif path.is_dir():
                resolved.extend(sorted(path.glob("*.json")))
            else:
                print(f"⚠️ Warning: Unsupported or missing path: {token}")

    return sorted(set(resolved))

# -----------------------------------------------------------
# Tree printing (unchanged)
# -----------------------------------------------------------
def print_tree(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"\n📖 {json_path.name}")

    for p_idx, part in enumerate(data.get("parts", []), 1):
        title = part.get("title") or part.get("part_id", f"Part {p_idx}")
        print(f"└── Part: {title}")

# -----------------------------------------------------------
# Counting core
# -----------------------------------------------------------
def count_title(title: Dict[str, Any]) -> Tuple[int, int, int]:
    paragraphs = title.get("paragraphs", [])
    p_count = len(paragraphs)
    w_count = 0
    l_count = 0

    for p in paragraphs:
        w_count += count_words(p.get("original_text"))
        if "lemmatized_text" in p:
            l_count += count_lemmas(p.get("lemmatized_text"))

    return p_count, w_count, l_count


def count_book(book: Dict[str, Any]) -> Tuple[int, int, int, int]:
    title_count = 0
    paragraph_count = 0
    word_count = 0
    lemma_count = 0

    titles = book.get("titles")

    if titles:
        for t in titles:
            title_count += 1
            pc, wc, lc = count_title(t)
            paragraph_count += pc
            word_count += wc
            lemma_count += lc
    else:
        paragraphs = book.get("paragraphs", [])
        paragraph_count += len(paragraphs)
        for p in paragraphs:
            word_count += count_words(p.get("original_text"))
            if "lemmatized_text" in p:
                lemma_count += count_lemmas(p.get("lemmatized_text"))

    return title_count, paragraph_count, word_count, lemma_count

# -----------------------------------------------------------
# Corpus analyzer
# -----------------------------------------------------------
def analyze_corpus(
    json_path: Path,
    show_parts: bool = False,
    show_books: bool = False,
    show_snippets: bool = False,
):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    print(f"\n📖 {json_path.name}")
    print(f"   📚 Corpus name: {data.get('corpus_name', 'Unknown')}")
    print(f"   🌍 Language: {data.get('language', 'Unknown')}")

    parts = data.get("parts", [])
    part_count = len(parts)

    book_count = 0
    title_count = 0
    paragraph_count = 0
    word_count = 0
    lemma_count = 0

    for part in parts:
        if show_parts:
            part_title = part.get("title") or part.get("part_id", "Untitled Part")
            print(f"  🧩 Part: {part_title}")

        for book in part.get("books", []):
            if isinstance(book, list):
                for sb in book:
                    book_count += 1
                    tc, pc, wc, lc = count_book(sb)
                    title_count += tc
                    paragraph_count += pc
                    word_count += wc
                    lemma_count += lc
            elif isinstance(book, dict):
                book_count += 1
                tc, pc, wc, lc = count_book(book)
                title_count += tc
                paragraph_count += pc
                word_count += wc
                lemma_count += lc

    print(f"\n   🔹 Parts: {part_count}")
    print(f"   🔹 Books: {book_count}")
    print(f"   🔹 Titles: {title_count}")
    print(f"   🔹 Paragraphs: {paragraph_count}")
    print(f"   🔹 Total words: {word_count:,}")
    print(f"   🔹 Total lemmas: {lemma_count:,}")

    if paragraph_count:
        print(f"   📊 Avg words per paragraph: {word_count / paragraph_count:.1f}")
        if lemma_count:
            print(f"   📊 Avg lemmas per paragraph: {lemma_count / paragraph_count:.1f}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze JSON legal corpora: structure, words, lemmas."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["./data/raw"],
        help="Path(s): file, directory, wildcard, or comma-separated list.",
    )
    parser.add_argument("--show_parts", action="store_true", help="Show list of parts")
    parser.add_argument("--show_books", action="store_true")
    parser.add_argument("--show_snippets", action="store_true")
    parser.add_argument("--tree", action="store_true")

    args = parser.parse_args()
    files = resolve_input_paths(args.paths)

    if not files:
        print("❌ No JSON files found.")
        sys.exit(1)

    if args.tree:
        print(f"\n🔍 TREE MODE: {len(files)} file(s)\n")
        for f in files:
            print_tree(f)
        sys.exit(0)

    print(f"\n🔍 Found {len(files)} JSON file(s) to analyze.\n")
    for f in files:
        analyze_corpus(
            f,
            show_parts=args.show_parts,
            show_books=args.show_books,
            show_snippets=args.show_snippets,
        )

    print("\n✅ Done!")