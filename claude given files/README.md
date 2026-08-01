# BAPS Document Translator (Local, Offline)

Translates English / Hindi / Gujarati documents (PDF, EPUB, TXT) into Bengali,
100% locally on your machine — no data leaves your laptop.

Built for: Dell Inspiron 14 Plus 7440, Intel Core Ultra 7 155H, 16GB RAM,
Intel Arc integrated graphics (CPU-only inference, no CUDA).

## 1. One-time setup

```bash
cd baps-translator
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

The two models you already downloaded:
- `ai4bharat/indictrans2-en-indic-1B` (English source)
- `ai4bharat/indictrans2-indic-indic-1B` (Hindi/Gujarati source)

`translator.py` loads these by name via HuggingFace `transformers`. If you
downloaded them to a custom local folder instead of the default HF cache,
update `EN_INDIC_MODEL` / `INDIC_INDIC_MODEL` in `config.py` to point to
that folder path instead of the HuggingFace repo name.

**Install the Bengali font** (for correct rendering in the output .docx):
download and install [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali)
on your machine (double-click the .ttf file → Install). Without it, Word
will substitute a fallback font.

## 2. Verify everything works before running a full book

```bash
python test_translation.py
```

This checks: models load correctly, language auto-detection works,
translation output is real Bengali (not garbled), and that numbers/glossary
terms (like "1801" or "Bhagwan Swaminarayan") survive translation intact.
**Do not skip this step** — it catches setup problems in 30 seconds instead
of 2 hours into a real book.

## 3. Add your own glossary terms

Edit `glossary.csv` — one line per term, format:
```
Source Term,Bengali Translation
```
Add character names, place names, and any BAPS-specific vocabulary you
want protected from mistranslation. Longer terms are matched first
automatically, so partial overlaps (e.g. "Swami" inside "Pramukh Swami
Maharaj") won't cause problems.

## 4. Run the app

```bash
streamlit run app.py
```

Opens automatically at http://localhost:8501. Upload a file, click
**Start Translation**, and let it run. On your CPU-only hardware, expect
roughly 20–60 seconds per page — for a full book, start it and let it run
in the background (you can close the browser tab; the terminal process
keeps going).

## 5. Resuming an interrupted translation

Re-upload the same file. The app detects the existing `*_progress.db`
file and offers **Resume**. Never delete the `.db` file mid-translation.

## Known limitations (read this so results match expectations)

- **No MT model is 100% accurate.** IndicTrans2 is the strongest available
  option for these language pairs, but idiomatic phrases, poetry, and
  ambiguous sentences can still translate imperfectly. Always have a
  Bengali speaker review output intended for publication.
- **Formatting is structure-preserving, not pixel-identical.** Bengali
  script takes more horizontal space than Gujarati/Hindi for the same
  content, so the output is a freshly laid-out document with the same
  chapter/heading/paragraph structure — not an exact visual clone of the
  original PDF.
- **Glossary protection is best-effort.** If a protected term's
  placeholder gets mangled by the model in unusual sentence structures,
  it will show up as `__GLOSSARY_N__` in the output — search for this
  string after large runs and fix manually if it appears.
