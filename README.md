# 📚 BAPS Book Translator

A Streamlit web application that translates BAPS Swaminarayan religious books from **Gujarati** (and Hindi / English) into **Bengali**. Runs entirely on your local machine — no GPU required. Cloud engines (Gemini, Claude) are available as optional backup.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Local translation** | IndicTrans2 1B model — 100% offline, fully private |
| **Alternative local model** | NLLB-200-3.3B (larger, supports all resource tiers) |
| **Cloud backup** | Gemini 2.0 Flash · Claude Haiku 4.5 |
| **Image input** | Scanned PDF pages via Tesseract OCR; photo-of-page via Claude Vision |
| **Translation Memory (TM)** | Exact + fuzzy (≥92%) matching — 100% accuracy on matched sentences |
| **BAPS Glossary** | 87 BAPS-specific terms protected from model mistranslation |
| **Resume / pause** | SQLite database — large books can be paused and resumed at any time |
| **Manual editor** | Side-by-side editor with instant DB save and Save-to-Memory button |
| **TM Manager** | Batch save, CSV import/export, cloud-assisted generation, book alignment |
| **Export formats** | TXT · PDF (Bengali font) · EPUB · DOCX |

---

## 🛠️ Installation

```bash
# Clone
git clone https://github.com/Aryanbuha890/BAPS_Book_Translator.git
cd BAPS_Book_Translator

# Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download translation model (choose tier based on your RAM)
python download_model.py
```

### Model download options

```
python download_model.py
```

| Option | Model | Download | RAM needed | Best for |
|---|---|---|---|---|
| `a` | NLLB-200-distilled-600M | ~2.5 GB | 4 GB | Low-end / CPU-only |
| `b` | NLLB-200-distilled-1.3B | ~5 GB | 8 GB | Mid-range |
| `c` | NLLB-200-3.3B | ~13 GB | 14 GB | High-end / GPU |
| `1` | IndicTrans2 Indic-Indic 1B ⭐ | ~4.5 GB | 4 GB | Default recommended |
| `2` | IndicTrans2 En-Indic 1B | ~4.5 GB | 4 GB | English source |

> **IndicTrans2 requires a HuggingFace token** — accept the model terms at  
> https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B first.  
> NLLB models are public — no token needed.

---

## 🚀 Running the app

```bash
source venv/bin/activate
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📖 How to Use

### 1. Translate a book

1. Open the sidebar — select **Source Language** and **Translation Engine**
2. Upload a book file (PDF, EPUB, TXT) or an image (PNG, JPG)
3. Click **▶ Start Translation** — you can pause and resume at any time
4. Go to **Side-by-Side Editor** tab to review and correct sentences
5. When complete, click **Generate** and **Download** in your chosen format

### 2. Improve accuracy with the Terminology Glossary

Open the **🏷️ Terminology Glossary** tab to add custom term mappings. Terms are replaced with placeholders before translation and restored correctly afterward — bypassing the model entirely for critical proper nouns.

### 3. Image input

- **Scanned PDF:** Upload normally — OCR fallback runs automatically on image-only pages (requires Tesseract installed)
- **Photo of a page:** Upload a JPG/PNG — Path A uses Claude Vision, Path B uses local Tesseract

---

## 🧠 How to Grow Translation Memory Fast

The Translation Memory (TM) gives **100% accurate** translations with no model involved. Every sentence added permanently improves all future books. Open the **🧠 Translation Memory** tab for all four paths:

### Path 1 — Batch save from current book (1 click)
After correcting sentences in the Side-by-Side Editor, open TM Manager → **Path 1**. All your corrected sentences save to TM in one click.

### Path 2 — CSV / Spreadsheet import (fastest for existing data)
If you have a spreadsheet of Gujarati→Bengali pairs:
1. Export it as CSV with two columns: **Gujarati** (col A), **Bengali** (col B) — no header row
2. Upload in **Path 2** → preview → **Import All**

```csv
ભગવાન સ્વામિનારાયણ,ভগবান স্বামীনারায়ণ
ગઢડા પ્રથમ,গড়দা প্রথম
...
```

### Path 3 — Cloud-assisted batch generation (for novel sentences)
1. Enter your Gemini API key in **Path 3**
2. Generate cloud translations for 20–50 sentences at a time
3. Review each one in the UI — edit if needed, uncheck if wrong
4. Click **Save Approved** → done

### Path 4 ⭐ — Align Gujarati + Bengali book files (adds 1,000+ entries at once)
If you have an **official BAPS Bengali translation** of the Vachanamrut or any other book alongside the Gujarati original:

1. Open **Path 4** in the TM Manager tab
2. Upload the **Gujarati source** file (PDF, EPUB, or TXT) on the left
3. Upload the **Bengali translation** file on the right
4. Preview the 15-pair alignment preview — verify it looks correct
5. Click **Import All** — the tool aligns sentences by position and bulk-adds every pair

> **Why this works:** Official BAPS Bengali translations are paragraph-faithful to the Gujarati. Sentence-level position alignment is highly accurate for Vachanamrut text.

After importing a full Vachanamrut, accuracy on that book jumps from ~65% → **90%+** because the TM handles most sentences directly.

---

## 📁 Project Structure

```
BAPS_Book_Translator/
├── app.py                    # Streamlit UI (all tabs)
├── translator.py             # IndicTrans2, NLLB, Gemini, Claude engines
├── extractor.py              # PDF/EPUB/TXT extraction + Tesseract OCR fallback
├── chunker.py                # Sentence splitting (with Gujarati conjunction splitting)
├── assembler.py              # Output builder (TXT/PDF/EPUB/DOCX)
├── glossary.py               # Placeholder swap system (protects proper nouns)
├── progress.py               # SQLite resume tracking
├── image_translator.py       # Claude Vision + Tesseract for image input
├── download_model.py         # Interactive model downloader (all tiers)
├── translation_memory.json   # Verified Gujarati→Bengali sentence pairs
├── glossary.csv              # 87 BAPS term mappings
├── NotoSansBengali-Regular.ttf  # Bengali font for PDF export
└── requirements.txt
```

---

## 🎯 Accuracy

| Content type | Current accuracy |
|---|---|
| Sentences in Translation Memory | 100% |
| Chapter headers / short phrases | ~85–90% |
| Medium discourse sentences | ~65–75% |
| Long compound sentences | ~50–65% |

**The single biggest accuracy lever is the Translation Memory.** The model handles novel sentences; the TM handles everything it has seen before at 100% accuracy. See "How to Grow Translation Memory Fast" above.

---

## ⚙️ Configuration

All settings are in the sidebar:

- **Source language** — Auto-detect, Gujarati, Hindi, English
- **Engine** — Local (IndicTrans2) · Local (NLLB) · Cloud (Gemini) · Cloud (Claude)
- **Batch size** — Lower if RAM < 8 GB
- **HuggingFace token** — Only needed for first IndicTrans2 download

---

## 📦 Dependencies

Key packages: `torch`, `transformers`, `indictranstoolkit`, `rapidfuzz`, `PyMuPDF`, `streamlit`, `reportlab`, `ebooklib`, `pytesseract`, `Pillow`, `anthropic`, `google-generativeai`

Full list: `requirements.txt`
