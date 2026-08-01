"""
translator.py

Loads AI4Bharat's IndicTrans2 models and translates text chunks to Bengali.

Correctness notes (this is the part people usually get wrong):
- IndicTrans2 CANNOT be called with raw text on a plain HF pipeline. It
  requires the official IndicTransToolkit `IndicProcessor` to (a) tag each
  sentence with its language, (b) normalize punctuation/quotes/numerals,
  (c) protect non-translatable spans (numbers, URLs, emails, dates) with
  placeholders during generation, and (d) restore them afterwards.
  Skipping this step is the #1 cause of "IndicTrans2 gives garbage output".
- English source text MUST use the en-indic checkpoint. The indic-indic
  checkpoint was never trained on English and will not translate it
  correctly.
"""

from functools import lru_cache
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

import config


def detect_source_lang(text: str) -> str:
    """
    Detect source language from a text sample using Unicode script ranges.
    Returns one of config.LANG_ENGLISH / LANG_HINDI / LANG_GUJARATI.
    Falls back to English if the script can't be identified (e.g. mostly
    numbers/punctuation) so the pipeline never crashes on ambiguous input.
    """
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    gujarati = len(re.findall(r"[\u0A80-\u0AFF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    counts = {
        config.LANG_HINDI: devanagari,
        config.LANG_GUJARATI: gujarati,
        config.LANG_ENGLISH: latin,
    }
    best_lang = max(counts, key=counts.get)
    if counts[best_lang] == 0:
        return config.LANG_ENGLISH
    return best_lang


def _checkpoint_for(src_lang: str) -> str:
    """Route to the correct model checkpoint for a given source language."""
    if src_lang == config.LANG_ENGLISH:
        return config.EN_INDIC_MODEL
    return config.INDIC_INDIC_MODEL


import os

# Dynamically resolve project root's 'models' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@lru_cache(maxsize=2)
def _load(checkpoint: str):
    """
    Load (and cache) a tokenizer + model from the local 'models' directory.
    """
    # Try local cache loading first. If files aren't found locally, fall back to online downloading.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, 
            trust_remote_code=True, 
            cache_dir=MODELS_DIR, 
            local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            cache_dir=MODELS_DIR,
            torch_dtype=torch.float32,
            local_files_only=True
        )
    except Exception:
        # Fallback to online loading if local copy is not cached
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, cache_dir=MODELS_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            cache_dir=MODELS_DIR,
            torch_dtype=torch.float32,
        )
    model = model.to(config.DEVICE)
    model.eval()
    return tokenizer, model


def translate_chunks(chunks: list[str], src_lang: str) -> list[str]:
    """
    Translate a list of text chunks from src_lang to Bengali.

    Args:
        chunks: list of source-language sentences/short paragraphs.
        src_lang: one of config.LANG_ENGLISH / LANG_HINDI / LANG_GUJARATI.

    Returns:
        list of Bengali translations, same order and length as `chunks`.
    """
    if not chunks:
        return []

    checkpoint = _checkpoint_for(src_lang)
    tokenizer, model = _load(checkpoint)
    ip = IndicProcessor()

    results: list[str] = []
    batch_size = config.BATCH_SIZE

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # Skip pure-whitespace/empty entries (paragraph-break markers from
        # chunker.py) without sending them through the model.
        real_indices = [j for j, c in enumerate(batch) if c.strip()]
        if not real_indices:
            results.extend(batch)  # preserve blank markers as-is
            continue

        real_texts = [batch[j] for j in real_indices]

        preprocessed = ip.preprocess_batch(
            real_texts, src_lang=src_lang, tgt_lang=config.TARGET_LANG
        )

        inputs = tokenizer(
            preprocessed,
            truncation=True,
            padding="longest",
            max_length=config.MAX_INPUT_TOKENS,
            return_tensors="pt",
        ).to(config.DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=config.MAX_INPUT_TOKENS,
                num_beams=config.NUM_BEAMS,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translated_real = ip.postprocess_batch(decoded, lang=config.TARGET_LANG)

        # Reassemble batch, putting blanks back where they were
        translated_batch = list(batch)
        for j, translated_text in zip(real_indices, translated_real):
            translated_batch[j] = translated_text

        results.extend(translated_batch)

    return results
