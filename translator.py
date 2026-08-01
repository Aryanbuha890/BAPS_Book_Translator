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

# Load BAPS Translation Memory (Translation Memory / Verified Sentences)
TM_PATH = "translation_memory.json"
_translation_memory = {}
if os.path.exists(TM_PATH):
    try:
        with open(TM_PATH, "r", encoding="utf-8") as f:
            raw_tm = json.load(f)
            # Normalize keys to standard spacing for robust matches
            for k, v in raw_tm.items():
                normalized_k = re.sub(r'\s+', ' ', k.strip())
                _translation_memory[normalized_k] = v
    except Exception as e:
        print(f"Warning: Failed to load translation memory: {e}")

def check_translation_memory(text: str) -> str:
    """
    Checks if the normalized text exists in the translation memory.
    If yes, returns the verified translation. Else, returns None.
    """
    normalized = re.sub(r'\s+', ' ', text.strip())
    # Try exact match
    if normalized in _translation_memory:
        return _translation_memory[normalized]
        
    # Try without trailing dots/punctuation
    normalized_no_punc = re.sub(r'[.!?।॥]\s*$', '', normalized).strip()
    for src, tgt in _translation_memory.items():
        src_no_punc = re.sub(r'[.!?।॥]\s*$', '', src).strip()
        if normalized_no_punc == src_no_punc:
            return tgt
            
    return None

def _load_local_model(model_name: str, hf_token: str = None):
    """
    Initializes and caches IndicTrans2 model and tokenizers on CPU dynamically.
    Loads on-demand depending on target checkpoint (indic-indic or en-indic).
    """
    global _local_tokenizers, _local_models, _local_processor, _local_device
    
    if model_name in _local_models:
        return
        
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        raise ImportError(
            "The local translation engine requires the `indictranstoolkit` package, which must be compiled "
            "from source using Microsoft Visual C++ Build Tools on Windows.\n\n"
            "If you do not have C++ Build Tools installed, please toggle the 'Translation Engine' in the left "
            "sidebar to either 'Cloud (Gemini)' or 'Cloud (Claude)'. They do not require any local compiler or model downloads."
        )

    if _local_device is None:
        _local_device = torch.device("cpu")
        
    if _local_processor is None:
        _local_processor = IndicProcessor(inference_stage="model")
        
    kwargs = {"trust_remote_code": True, "cache_dir": "models"}
    if hf_token and hf_token.strip():
        kwargs["token"] = hf_token.strip()
    
    print(f"Loading local checkpoint '{model_name}' on CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(_local_device)
        
        # Cache model and tokenizer
        _local_tokenizers[model_name] = tokenizer
        _local_models[model_name] = model
        print(f"✓ Loaded '{model_name}' successfully.")
    except Exception as first_error:
        # Fallback to local files only in case of connection or token authentication issues
        kwargs["local_files_only"] = True
        if "token" in kwargs:
            del kwargs["token"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(_local_device)
            
            _local_tokenizers[model_name] = tokenizer
            _local_models[model_name] = model
            print(f"✓ Loaded '{model_name}' from offline local files.")
        except Exception as fallback_error:
            raise OSError(
                f"Failed to load local model '{model_name}'. Please ensure you have downloaded "
                f"the weights using 'download_model.py' first.\n\n"
                f"Details:\nFirst attempt error: {first_error}\nOffline fallback error: {fallback_error}"
            )

def clean_translated_text(text: str) -> str:
    """
    Cleans up common translation artifacts like duplicate repeating symbols (=, -) 
    introduced by tokenizer limits or model glitches, and corrects spelling variations of spiritual terms.
    """
    # Remove repeating equals signs, spaces, or dashes
    text = re.sub(r'={2,}', '', text)
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    # BAPS spiritual spelling corrections
    corrections = {
        "ভক্সনামৃতমে": "বচনামৃতে",
        "ভক্সনামৃত": "বচনামৃত",
        "বাক্যামৃত্রের": "বচনামৃতের",
        "বাক্যামৃত": "বচনামৃত",
        "শ্রীজি মহারাজ": "শ্রীজী মহারাজ",
        "নিশ্চলানন্দ স্বামী": "নিষ্কুলানন্দ স্বামী",
        "স্বয়ান্ হরি": "স্বয়ং হরি",
        "স্বয় হরি": "স্বয়ং হরি",
        "স্বয়হান হরি": "স্বয়ং হরি",
        "সতপুরুষ": "সৎপুরুষ",
        "মহান্ত স্বামী": "মহন্ত স্বামী",
        "তীতর": "তীর্থ",
        "মায়েশ": "মহন্ত"
    }
    
    for typo, correct in corrections.items():
        text = text.replace(typo, correct)
        
    return text.strip()

def translate_chunks_local(
    chunks: list[str], 
    src_lang: str = "guj_Gujr", 
    tgt_lang: str = "ben_Beng", 
    batch_size: int = 4, 
    hf_token: str = None
) -> list[str]:
    """
    Translates a list of text chunks using local IndicTrans2 1B models.
    Selects the correct 1B model (indic-indic or en-indic) dynamically.
    """
    if src_lang == "eng_Latn":
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    else:
        model_name = "ai4bharat/indictrans2-indic-indic-1B"
        
    _load_local_model(model_name, hf_token=hf_token)
    tokenizer = _local_tokenizers[model_name]
    model = _local_models[model_name]
    
    results = []
    
    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Check for translation memory hits and empty chunks
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
            
        # IndicTrans2 preprocessing
        preprocessed = _local_processor.preprocess_batch(non_empty_texts, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize and run model
        inputs = tokenizer(
            preprocessed, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(_local_device)
        
        with torch.no_grad():
            # Add repetition_penalty to avoid repeating character loops (like = = = = or word loops)
            outputs = model.generate(
                **inputs, 
                num_beams=5, 
                max_length=256,
                repetition_penalty=1.2
            )
            
        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        postprocessed = _local_processor.postprocess_batch(decoded, lang=tgt_lang)
        
        # Clean postprocessed text
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
    glossary: dict = None
) -> list[str]:
    """
    Translates chunks using Gemini 1.5 Flash. Groups sentences for speed.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Map language tag to readable name
    lang_names = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "ben_Beng": "Bengali"}
    src_name = lang_names.get(src_lang, "Gujarati")
    tgt_name = lang_names.get(tgt_lang, "Bengali")
    
    # 1. Resolve Translation Memory hits first
    final = [None] * len(chunks)
    for idx, val in enumerate(chunks):
        if not val.strip():
            final[idx] = ""
            continue
        mem_match = check_translation_memory(val)
        if mem_match:
            final[idx] = mem_match
            
    # Find remaining items
    non_empty = [(idx, chunks[idx]) for idx in range(len(chunks)) if final[idx] is None]
    if not non_empty:
        return final
        
    indices, texts = zip(*non_empty)
    glossary_str = json.dumps(glossary, ensure_ascii=False) if glossary else "{}"
    
    prompt = (
        f"You are a professional book translator. Translate the following {src_name} sentences to {tgt_name}. "
        "Maintain the tone, style, and paragraph continuity.\n"
        f"Apply these terminology glossary replacements if applicable: {glossary_str}\n\n"
        "Return ONLY a JSON array of strings of the exact same length in the same index order. "
        "Do not include markdown tags (like ```json), comments, or code formatting. Just return a raw JSON array.\n\n"
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
            print("Gemini batch response length mismatch or incorrect format. Falling back to single translations.")
    except Exception as e:
        print(f"Gemini batch translation error: {e}. Falling back to single translations.")
        pass

    # Fallback execution for remaining items
    for idx, text in non_empty:
        p = f"Translate the following {src_name} text to {tgt_name}. Output ONLY the translation:\n\n{text}"
        try:
            r = model.generate_content(p)
            final[idx] = r.text.strip()
        except Exception as e:
            print(f"Gemini single translation error for '{text}': {e}")
            final[idx] = "[Translation Error]"
    return final

def translate_chunks_claude(
    chunks: list[str], 
    api_key: str, 
    src_lang: str = "guj_Gujr", 
    tgt_lang: str = "ben_Beng", 
    glossary: dict = None
) -> list[str]:
    """
    Translates chunks using Claude API.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    # Map language tag to readable name
    lang_names = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "ben_Beng": "Bengali"}
    src_name = lang_names.get(src_lang, "Gujarati")
    tgt_name = lang_names.get(tgt_lang, "Bengali")
    
    # 1. Resolve Translation Memory hits first
    final = [None] * len(chunks)
    for idx, val in enumerate(chunks):
        if not val.strip():
            final[idx] = ""
            continue
        mem_match = check_translation_memory(val)
        if mem_match:
            final[idx] = mem_match
            
    # Find remaining items
    non_empty = [(idx, chunks[idx]) for idx in range(len(chunks)) if final[idx] is None]
    if not non_empty:
        return final
        
    indices, texts = zip(*non_empty)
    glossary_str = json.dumps(glossary, ensure_ascii=False) if glossary else "{}"
    
    prompt = (
        f"Translate these {src_name} sentences to {tgt_name}. Keep the exact order and count. "
        f"Glossary replacements: {glossary_str}\n\n"
        "Output ONLY a valid JSON list of strings (no markdown blocks, no prefix/suffix comments):\n"
        f"{json.dumps(list(texts), ensure_ascii=False)}"
    )
    
    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            system="You are a translator that ONLY outputs a valid JSON list of translated strings.",
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

    # Fallback execution
    for idx, text in non_empty:
        try:
            r = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Translate this {src_name} text to {tgt_name}. Output only translation:\n\n{text}"}]
            )
            final[idx] = r.content[0].text.strip()
        except:
            final[idx] = "[Translation Error]"
    return final

def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """
    Applies glossary regex substitutions to raw text before translation.
    """
    if not glossary:
        return text
    for orig, trans in glossary.items():
        # Match words (using word boundary pattern matching)
        text = re.sub(rf'\b{re.escape(orig)}\b', trans, text)
    return text
