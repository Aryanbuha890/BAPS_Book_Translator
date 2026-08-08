import os
import re
import json
import torch
from tqdm import tqdm
from progress import ProgressDB

# Lazy loading dictionaries to hold both English-Indic and Indic-Indic checkpoints
_local_tokenizers = {}
_local_models = {}
_local_processor = None
_local_device = None

import unicodedata

def clean_and_normalize(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
    return text

# Load BAPS Translation Memory at startup
TM_PATH = "translation_memory.json"
_translation_memory = {}
if os.path.exists(TM_PATH):
    try:
        with open(TM_PATH, "r", encoding="utf-8") as f:
            raw_tm = json.load(f)
            for k, v in raw_tm.items():
                normalized_k = clean_and_normalize(k)
                _translation_memory[normalized_k] = v
    except Exception as e:
        print(f"Warning: Failed to load translation memory: {e}")

def reload_translation_memory():
    """Reloads translation_memory.json into the in-memory cache. Call after adding new entries."""
    global _translation_memory
    _translation_memory = {}
    if os.path.exists(TM_PATH):
        try:
            with open(TM_PATH, "r", encoding="utf-8") as f:
                raw_tm = json.load(f)
                for k, v in raw_tm.items():
                    _translation_memory[clean_and_normalize(k)] = v
        except Exception as e:
            print(f"Warning: Failed to reload translation memory: {e}")

def save_to_translation_memory(original_text: str, translated_text: str):
    """
    Appends a verified Gujarati→Bengali pair to translation_memory.json
    and refreshes the in-memory cache.
    """
    original_text = original_text.strip()
    translated_text = translated_text.strip()
    if not original_text or not translated_text:
        return
    existing = {}
    if os.path.exists(TM_PATH):
        try:
            with open(TM_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing[original_text] = translated_text
    with open(TM_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    reload_translation_memory()

def check_translation_memory(text: str) -> str:
    """
    Checks translation memory: exact match first, then punctuation-stripped, then fuzzy (≥92%).
    Returns the verified translation or None.
    """
    normalized = clean_and_normalize(text)

    # 1. Exact match
    if normalized in _translation_memory:
        return _translation_memory[normalized]

    # 2. Punctuation-stripped match
    normalized_no_punc = re.sub(r'[.!?।॥]\s*$', '', normalized).strip()
    for src, tgt in _translation_memory.items():
        src_no_punc = re.sub(r'[.!?।॥]\s*$', '', src).strip()
        if normalized_no_punc == src_no_punc:
            return tgt

    # 3. Fuzzy match — only worthwhile for sentences of reasonable length
    if len(normalized) >= 10 and _translation_memory:
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_val = None
            for src, tgt in _translation_memory.items():
                score = fuzz.ratio(normalized, src)
                if score > best_score:
                    best_score = score
                    best_val = tgt
            if best_score >= 92:
                return best_val
        except ImportError:
            pass

    return None

def _load_local_model(model_name: str, hf_token: str = None):
    """
    Initializes and caches IndicTrans2 model and tokenizer on CPU (or CUDA if available).
    """
    global _local_tokenizers, _local_models, _local_processor, _local_device

    if model_name in _local_models:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    if _local_device is None:
        if torch.cuda.is_available():
            _local_device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _local_device = torch.device("mps")
        else:
            _local_device = torch.device("cpu")
        print(f"Using device: {_local_device}")

    if _local_processor is None:
        _local_processor = IndicProcessor()

    # Resolve local path: snapshot_download saves to models/<short-name>/
    short_name = model_name.split("/")[-1]  # e.g. "indictrans2-indic-indic-1B"
    local_dir = os.path.join("models", short_name)
    load_path = local_dir if os.path.isdir(local_dir) else model_name

    base_kwargs = {"trust_remote_code": True}
    if load_path == model_name:
        # Remote load — use HuggingFace cache
        base_kwargs["cache_dir"] = "models"
        if hf_token and hf_token.strip():
            base_kwargs["token"] = hf_token.strip()

    print(f"Loading local checkpoint from '{load_path}' on {_local_device.type.upper()}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(load_path, **base_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(load_path, **base_kwargs).to(_local_device)
        _local_tokenizers[model_name] = tokenizer
        _local_models[model_name] = model
        print(f"✓ Loaded '{model_name}' successfully.")
    except Exception as first_error:
        if load_path != model_name:
            raise OSError(
                f"Failed to load model from local path '{load_path}'.\n"
                f"Error: {first_error}\n\n"
                "Re-run 'python download_model.py' to re-download the weights."
            )
        # Offline fallback for remote path
        fallback_kwargs = {**base_kwargs, "local_files_only": True}
        fallback_kwargs.pop("token", None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **fallback_kwargs)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **fallback_kwargs).to(_local_device)
            _local_tokenizers[model_name] = tokenizer
            _local_models[model_name] = model
            print(f"✓ Loaded '{model_name}' from offline cache.")
        except Exception as fallback_error:
            raise OSError(
                f"Failed to load '{model_name}'.\n"
                f"First attempt: {first_error}\nOffline fallback: {fallback_error}\n\n"
                "Run 'python download_model.py' to download weights."
            )

def clean_translated_text(text: str) -> str:
    """
    Removes model artifacts and applies BAPS-specific Bengali corrections.

    Three categories of corrections:
    1. Spelling / transliteration errors (BAPS proper nouns)
    2. Honorific verb forms — the model defaults to informal Bengali;
       in BAPS spiritual text the subject is almost always God or a Sant,
       so respectful forms are correct throughout.
    3. Honorific pronouns — same reasoning.
    """
    text = re.sub(r'={2,}', '', text)
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # ── 1. Spelling corrections (longest-first to avoid substring clobbering) ──
    spelling = [
        ("বাক্যামৃত্রের",         "বচনামৃতের"),
        ("নিশ্চলানন্দ স্বামী",   "নিষ্কুলানন্দ স্বামী"),
        ("মহান্তস্বামী",          "মহন্ত স্বামী"),
        ("মহান্ত স্বামী",         "মহন্ত স্বামী"),
        ("শাস্ত্রীজি মহারাজ",    "শাস্ত্রীজী মহারাজ"),
        ("স্বয়ান্ হরি",          "স্বয়ং হরি"),
        ("স্বয়হান হরি",          "স্বয়ং হরি"),
        ("স্বয়ান হরি",           "স্বয়ং হরি"),
        ("শ্রীজি মহারাজ",        "শ্রীজী মহারাজ"),
        ("ভক্সনামৃতমে",          "বচনামৃতে"),
        ("ভক্সনামৃত",            "বচনামৃত"),
        ("বাক্যামৃত",            "বচনামৃত"),
        ("বচনামৃতম",             "বচনামৃত"),
        ("সদ্পুরুষ",             "সৎপুরুষ"),
        ("সতপুরুষ",              "সৎপুরুষ"),
        ("শাস্ত্রীজি",           "শাস্ত্রীজী"),
        ("স্বয় হরি",             "স্বয়ং হরি"),
        ("স্বয়ান",               "স্বয়ং"),
        ("মায়েশ",               "মহন্ত"),
        ("তীতর",                 "তীর্থ"),
        ("দর্শণ",                "দর্শন"),
        # Model sometimes uses English "heart" — replace with Bengali
        ("হার্টের",              "হৃদয়ের"),
        ("হার্ট",                "হৃদয়"),
    ]

    for typo, correct in spelling:
        text = re.sub(
            r'(?<!\w)' + re.escape(typo) + r'(?!\w)',
            correct, text, flags=re.UNICODE
        )

    # ── 2. Honorific verb forms ────────────────────────────────────────────────
    # In BAPS spiritual text, subjects are God or a Sant — respectful forms required.
    # Only replace at clause/sentence boundaries (followed by punctuation or space+Bengali)
    # to avoid false matches inside longer words.
    # Pattern: informal ending → respectful ending, only when followed by [।.,;) ] or end-of-string
    _boundary = r'(?=[।.,;)\s]|$)'

    verb_fixes = [
        # Perfect tense: -েছে → -েছেন
        (r'বলেছে' + _boundary,      'বলেছেন'),
        (r'করেছে' + _boundary,      'করেছেন'),
        (r'দিয়েছে' + _boundary,    'দিয়েছেন'),
        (r'নিয়েছে' + _boundary,    'নিয়েছেন'),
        (r'দেখেছে' + _boundary,     'দেখেছেন'),
        (r'এসেছে' + _boundary,      'এসেছেন'),
        (r'গিয়েছে' + _boundary,    'গিয়েছেন'),
        (r'রেখেছে' + _boundary,     'রেখেছেন'),
        (r'শিখিয়েছে' + _boundary,  'শিখিয়েছেন'),
        # Present continuous: -ছে → -ছেন
        (r'করছে' + _boundary,       'করছেন'),
        (r'দিচ্ছে' + _boundary,     'দিচ্ছেন'),
        (r'দেখছে' + _boundary,      'দেখছেন'),
        (r'বলছে' + _boundary,       'বলছেন'),
        (r'থাকছে' + _boundary,      'থাকছেন'),
        (r'চলছে' + _boundary,       'চলছেন'),
        # Simple present: -ে → -েন  (more targeted to avoid false matches)
        (r'করে' + _boundary,        'করেন'),
        (r'দেয়' + _boundary,        'দেন'),
        (r'বলে' + _boundary,        'বলেন'),
    ]

    for pattern, replacement in verb_fixes:
        text = re.sub(pattern, replacement, text, flags=re.UNICODE)

    # ── 3. Honorific pronouns ─────────────────────────────────────────────────
    # 'তার' (informal his/her) → 'তাঁর' (respectful) — only when standalone word
    # Careful: must not match 'তারা' (they) or 'তারপর' (then)
    text = re.sub(r'(?<!\w)তার(?!\w)', 'তাঁর', text, flags=re.UNICODE)
    # 'তিনি' is already respectful; 'সে' → 'তিনি' is riskier, skip for now

    return text.strip()

NLLB_VARIANTS = {
    "nllb-600M":  "facebook/nllb-200-distilled-600M",
    "nllb-1.3B":  "facebook/nllb-200-distilled-1.3B",
    "nllb-3.3B":  "facebook/nllb-200-3.3B",
}

def _resolve_nllb_model() -> tuple[str, str]:
    """Returns (local_short_name, hf_repo_id) for the first NLLB variant found on disk."""
    for short, repo in [
        ("nllb-200-3.3B",            "facebook/nllb-200-3.3B"),
        ("nllb-200-distilled-1.3B",  "facebook/nllb-200-distilled-1.3B"),
        ("nllb-200-distilled-600M",  "facebook/nllb-200-distilled-600M"),
    ]:
        if os.path.isdir(os.path.join("models", short)):
            return short, repo
    return "nllb-200-3.3B", "facebook/nllb-200-3.3B"  # default

def translate_chunks_nllb(
    chunks: list[str],
    src_lang: str = "guj_Gujr",
    tgt_lang: str = "ben_Beng",
    batch_size: int = 4,
    hf_token: str = None
) -> list[str]:
    """
    Translates using Meta's NLLB-200 model (auto-selects the largest downloaded variant).
    Works directly with Gujarati script — no intermediate Devanagari conversion.
    Higher accuracy than IndicTrans2 1B on complex literary sentences.
    """
    short_name, hf_name = _resolve_nllb_model()
    model_name = hf_name
    short_name = short_name
    local_dir = os.path.join("models", short_name)
    load_path = local_dir if os.path.isdir(local_dir) else model_name

    global _local_tokenizers, _local_models, _local_device

    if _local_device is None:
        if torch.cuda.is_available():
            _local_device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _local_device = torch.device("mps")
        else:
            _local_device = torch.device("cpu")

    if model_name not in _local_models:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"Loading NLLB-200-3.3B from '{load_path}' on {_local_device.type.upper()}...")
        kwargs = {"trust_remote_code": True}
        if load_path == model_name:
            kwargs["cache_dir"] = "models"
            if hf_token:
                kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(load_path, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(load_path, **kwargs).to(_local_device)
        _local_tokenizers[model_name] = tokenizer
        _local_models[model_name] = model
        print("✓ NLLB-200-3.3B loaded.")

    tokenizer = _local_tokenizers[model_name]
    model = _local_models[model_name]

    # NLLB uses forced_bos_token_id to specify target language
    tgt_lang_token = tokenizer.convert_tokens_to_ids(tgt_lang)
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_results = [""] * len(batch)
        non_empty_indices, non_empty_texts = [], []

        for idx, text in enumerate(batch):
            if not text.strip():
                continue
            mem_match = check_translation_memory(text)
            if mem_match:
                batch_results[idx] = mem_match
            else:
                non_empty_indices.append(idx)
                non_empty_texts.append(text)

        if not non_empty_texts:
            results.extend(batch_results)
            continue

        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_local_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_token,
                num_beams=5,
                max_length=512,
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = [clean_translated_text(t) for t in decoded]

        for local_idx, orig_idx in enumerate(non_empty_indices):
            batch_results[orig_idx] = decoded[local_idx]

        results.extend(batch_results)

    return results


def translate_chunks_local(
    chunks: list[str],
    src_lang: str = "guj_Gujr",
    tgt_lang: str = "ben_Beng",
    batch_size: int = 4,
    hf_token: str = None
) -> list[str]:
    """
    Translates a list of text chunks using local IndicTrans2 1B models.
    Uses the real IndicTransToolkit for preprocessing/postprocessing.
    """
    if src_lang == "eng_Latn":
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    else:
        model_name = "ai4bharat/indictrans2-indic-indic-1B"

    _load_local_model(model_name, hf_token=hf_token)
    tokenizer = _local_tokenizers[model_name]
    model = _local_models[model_name]

    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Separate TM hits and empty strings from chunks that need translation
        non_empty_indices = []
        non_empty_texts = []
        batch_results = [""] * len(batch)

        for idx, text in enumerate(batch):
            if not text.strip():
                continue
            mem_match = check_translation_memory(text)
            if mem_match:
                batch_results[idx] = mem_match
            else:
                non_empty_indices.append(idx)
                non_empty_texts.append(text)

        if not non_empty_texts:
            results.extend(batch_results)
            continue

        # IndicTrans2 preprocessing via real IndicTransToolkit
        preprocessed = _local_processor.preprocess_batch(non_empty_texts, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            preprocessed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(_local_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=5,
                max_length=512,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )

        # Decode without the deprecated as_target_tokenizer() context manager
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        postprocessed = _local_processor.postprocess_batch(decoded, lang=tgt_lang)
        postprocessed = [clean_translated_text(t) for t in postprocessed]

        for local_idx, orig_idx in enumerate(non_empty_indices):
            batch_results[orig_idx] = postprocessed[local_idx]

        results.extend(batch_results)

    return results

def translate_chunks_gemini(
    chunks: list[str],
    api_key: str,
    src_lang: str = "guj_Gujr",
    tgt_lang: str = "ben_Beng",
    glossary: dict = None,
    prev_context: str = None
) -> list[str]:
    """
    Translates chunks using Gemini 2.0 Flash. Optionally prepends previous sentence as context.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    lang_names = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "ben_Beng": "Bengali"}
    src_name = lang_names.get(src_lang, "Gujarati")
    tgt_name = lang_names.get(tgt_lang, "Bengali")

    final = [None] * len(chunks)
    for idx, val in enumerate(chunks):
        if not val.strip():
            final[idx] = ""
            continue
        mem_match = check_translation_memory(val)
        if mem_match:
            final[idx] = mem_match

    non_empty = [(idx, chunks[idx]) for idx in range(len(chunks)) if final[idx] is None]
    if not non_empty:
        return final

    indices, texts = zip(*non_empty)
    glossary_str = json.dumps(glossary, ensure_ascii=False) if glossary else "{}"

    context_line = ""
    if prev_context and prev_context.strip():
        context_line = f"Previous sentence (context only, do not translate): {prev_context}\n\n"

    prompt = (
        f"You are a professional book translator specialising in BAPS Swaminarayan religious texts. "
        f"Translate the following {src_name} sentences to {tgt_name}, "
        "preserving the devotional tone and formal register of the original.\n"
        f"Apply these terminology mappings: {glossary_str}\n\n"
        f"{context_line}"
        "Return ONLY a JSON array of strings of the exact same length in the same index order. "
        "No markdown, no comments, just a raw JSON array.\n\n"
        f"Input: {json.dumps(list(texts), ensure_ascii=False)}"
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        translated = json.loads(response.text)
        if isinstance(translated, list) and len(translated) == len(texts):
            for idx, text in zip(indices, translated):
                final[idx] = text.strip()
            return final
        else:
            print("Gemini batch response length mismatch. Falling back to single translations.")
    except Exception as e:
        print(f"Gemini batch error: {e}. Falling back to single translations.")

    for idx, text in non_empty:
        p = f"Translate the following {src_name} text to {tgt_name}. Output ONLY the translation:\n\n{text}"
        try:
            r = model.generate_content(p)
            final[idx] = r.text.strip()
        except Exception as e:
            print(f"Gemini single translation error: {e}")
            final[idx] = "[Translation Error]"
    return final

def translate_chunks_claude(
    chunks: list[str],
    api_key: str,
    src_lang: str = "guj_Gujr",
    tgt_lang: str = "ben_Beng",
    glossary: dict = None,
    prev_context: str = None
) -> list[str]:
    """
    Translates chunks using Claude API (backup engine).
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    lang_names = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "ben_Beng": "Bengali"}
    src_name = lang_names.get(src_lang, "Gujarati")
    tgt_name = lang_names.get(tgt_lang, "Bengali")

    final = [None] * len(chunks)
    for idx, val in enumerate(chunks):
        if not val.strip():
            final[idx] = ""
            continue
        mem_match = check_translation_memory(val)
        if mem_match:
            final[idx] = mem_match

    non_empty = [(idx, chunks[idx]) for idx in range(len(chunks)) if final[idx] is None]
    if not non_empty:
        return final

    indices, texts = zip(*non_empty)
    glossary_str = json.dumps(glossary, ensure_ascii=False) if glossary else "{}"

    context_line = ""
    if prev_context and prev_context.strip():
        context_line = f"Previous sentence (context only, do not translate): {prev_context}\n\n"

    prompt = (
        f"Translate these {src_name} sentences to {tgt_name}. Keep the exact order and count.\n"
        f"Glossary: {glossary_str}\n\n"
        f"{context_line}"
        "Output ONLY a valid JSON list of strings (no markdown, no prefix/suffix):\n"
        f"{json.dumps(list(texts), ensure_ascii=False)}"
    )

    system_prompt = (
        "You are an expert literary translator specialising in BAPS Swaminarayan religious texts. "
        "Translate Gujarati to Bengali preserving the devotional tone, spiritual vocabulary, and "
        "formal register of the original. Key terms (Vachanamrut, Maharaj, Sant, Swaminarayan, "
        "Gunatit, Satpurush) must use their standard Bengali equivalents from the glossary provided. "
        "Output ONLY a valid JSON list of translated strings, same count and order as input."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        text_out = response.content[0].text.strip()
        if text_out.startswith("```"):
            text_out = re.sub(r"^```(json)?\n|```$", "", text_out, flags=re.MULTILINE)
        translated = json.loads(text_out)
        if isinstance(translated, list) and len(translated) == len(texts):
            for idx, text in zip(indices, translated):
                final[idx] = text.strip()
            return final
    except Exception:
        pass

    for idx, text in non_empty:
        try:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Translate this {src_name} text to {tgt_name}. Output only translation:\n\n{text}"}]
            )
            final[idx] = r.content[0].text.strip()
        except Exception:
            final[idx] = "[Translation Error]"
    return final

def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """
    Applies glossary substitutions using Unicode-aware lookarounds.
    Note: the main translation pipeline uses glossary.py swap_in/swap_out instead.
    This function is kept for any pre-translation text preprocessing use cases.
    """
    if not glossary:
        return text
    # Sort longest first to prevent shorter sub-terms stealing matches
    for orig in sorted(glossary.keys(), key=len, reverse=True):
        trans = glossary[orig]
        # Use lookarounds instead of \b — \b only works on ASCII word characters
        text = re.sub(
            r'(?<!\w)' + re.escape(orig) + r'(?!\w)',
            trans,
            text,
            flags=re.UNICODE
        )
    return text
