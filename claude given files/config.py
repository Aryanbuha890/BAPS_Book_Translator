"""
Central configuration for the BAPS document translator.
Edit values here rather than scattering constants across files.
"""

# --- Model checkpoints (already downloaded locally by the user) ---
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDIC_INDIC_MODEL = "ai4bharat/indictrans2-indic-indic-1B"

# --- FLORES-style language codes used by IndicTrans2 ---
LANG_ENGLISH = "eng_Latn"
LANG_HINDI = "hin_Deva"
LANG_GUJARATI = "guj_Gujr"
LANG_BENGALI = "ben_Beng"

TARGET_LANG = LANG_BENGALI

# --- Hardware tuning ---
# Dell Inspiron 14 Plus 7440: Intel Core Ultra 7 155H, 16GB RAM, Intel Arc
# integrated graphics (NOT CUDA-capable) -> inference runs on CPU.
DEVICE = "cpu"
BATCH_SIZE = 4          # keep small on CPU-only 16GB systems; raise to 8 if RAM allows
MAX_INPUT_TOKENS = 256  # per-chunk cap sent to the model
NUM_BEAMS = 5            # higher = better quality, slower. 5 is IndicTrans2's recommended default.

# --- Chunking ---
MAX_CHUNK_CHARS = 400

# --- Glossary ---
GLOSSARY_CSV_PATH = "glossary.csv"

# --- Output ---
OUTPUT_DIR = "output"
