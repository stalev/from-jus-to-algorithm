# Data folder

## Folder structure
- raw/        Source texts as collected (JSON), see SOURCES.md
- reference/  Curated reference files (e.g., anchors) produced by the authors

### Raw data files (`raw/`)
- `raw/roman.json` — Corpus Iuris Civilis (Justinian, 533), Books I–XII (Latin)
- `raw/napoleon.json` — Napoleonic codes (French): Code civil (1804), Code de procédure civile (1806), Code de commerce (1807)
- `raw/eu.json` — EU Digital Acquis via EUR-Lex ELI (directives + regulations listed in `SOURCES.md`)

Each JSON file is a curated snapshot of the source text(s) as collected at the time of the study.
See `SOURCES.md` for provenance (URLs) and reuse notes.

### Reference files (`reference/`)
- `reference/alignment_anchors.json` — alignment anchor lexicon (curated/selected by the authors) used for cross-epoch / cross-language alignment
- `reference/legal_anchors.json` — domain/legal anchor lexicon used for analysis (curated/selected by the authors)

These reference files are original research artifacts (curation/selection) and should be cited with the project DOI.

## What this data is
This folder contains curated/structured versions of legal texts used in the project:
- Latin: Corpus Iuris Civilis (Justinian, 533)
- French: Napoleonic codes (1804–1807) via Wikisource
- EU: selected directives and regulations via EUR-Lex (ELI)

## What we did (curation / preprocessing)
- Collected the source texts from the URLs listed in `SOURCES.md`
- Converted texts into a unified JSON representation (document → sections/articles/paragraphs where applicable)
- Normalized encoding and formatting (whitespace, punctuation normalization where needed)
- Added corpus metadata (epoch/jurisdiction/document identifiers)
- Preprocessed texts per language (`scripts/preprocess_la.py`, `preprocess_fr.py`, `preprocess_en.py`) and extracted lemma vocabularies (`scripts/extract_lemmas.py`)
- Trained embedding models (`scripts/train_models.py`) and performed alignment (`scripts/align_models.py`) using curated anchors in `data/reference/`

## How to cite
Preferred: cite the Zenodo release DOI (code + data snapshot).
Until a DOI is available, cite the GitHub repository and commit hash.

- Software & data snapshot (Zenodo DOI): <TO BE FILLED AFTER FIRST RELEASE>
- Preprint: <SSRN link/DOI>

## Licensing
- See SOURCES.md for third-party sources and their notices.
- See DATA_LICENSE.txt for how reuse works for the curated dataset vs underlying texts.