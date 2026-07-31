import os
import re
import json
import torch
from tqdm import tqdm
from progress import ProgressDB

# Lazy loading variables for local model to save start-up overhead
_local_tokenizer = None
_local_model = None
_local_processor = None
_local_device = None

def _load_local_model(hf_token: str = None):
    """
    Initializes and caches IndicTrans2 model and tokenizers on CPU.
    """
    global _local_tokenizer, _local_model, _local_processor, _local_device
    if _local_model is not None:
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

    model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
    _local_device = torch.device("cpu")
    
    kwargs = {"trust_remote_code": True, "cache_dir": "models"}
    if hf_token and hf_token.strip():
        kwargs["token"] = hf_token.strip()
    
    try:
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(_local_device)
    except Exception as first_error:
        # Fallback to local files only in case of connection or token authentication issues
        kwargs["local_files_only"] = True
        if "token" in kwargs:
            del kwargs["token"]
        try:
            _local_tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(_local_device)
        except Exception as fallback_error:
            raise OSError(
                f"Failed to load the model. Ensure you have entered a valid token or that "
                f"the model was downloaded successfully.\n\n"
                f"Details:\nFirst attempt error: {first_error}\nOffline fallback error: {fallback_error}"
            )
            
    _local_processor = IndicProcessor(inference_stage="model")

def translate_chunks_local(
    chunks: list[str], 
    src_lang: str = "guj_Gujr", 
    tgt_lang: str = "ben_Beng", 
    batch_size: int = 8, 
    hf_token: str = None
) -> list[str]:
    """
    Translates a list of text chunks using local IndicTrans2.
    """
    _load_local_model(hf_token=hf_token)
    results = []
    
    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Check for empty chunks (paragraph break markers)
        non_empty_indices = [idx for idx, text in enumerate(batch) if text.strip()]
        non_empty_texts = [batch[idx] for idx in non_empty_indices]
        
        batch_results = [""] * len(batch)
        if not non_empty_texts:
            results.extend(batch_results)
            continue
            
        # IndicTrans2 preprocessing
        preprocessed = _local_processor.preprocess_batch(non_empty_texts, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize and run model
        inputs = _local_tokenizer(
            preprocessed, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(_local_device)
        
        with torch.no_grad():
            outputs = _local_model.generate(**inputs, num_beams=5, max_length=256)
            
        with _local_tokenizer.as_target_tokenizer():
            decoded = _local_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        postprocessed = _local_processor.postprocess_batch(decoded, lang=tgt_lang)
        
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
    
    # Filter empty items
    non_empty = [(idx, val) for idx, val in enumerate(chunks) if val.strip()]
    if not non_empty:
        return [""] * len(chunks)
        
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
            final = [""] * len(chunks)
            for idx, text in zip(indices, translated):
                final[idx] = text.strip()
            return final
    except Exception as e:
        # Fallback to single-chunk translations if batch fails
        pass

    # Fallback execution
    final = [""] * len(chunks)
    for idx, text in non_empty:
        p = f"Translate the following {src_name} text to {tgt_name}. Output ONLY the translation:\n\n{text}"
        try:
            r = model.generate_content(p)
            final[idx] = r.text.strip()
        except:
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
    
    non_empty = [(idx, val) for idx, val in enumerate(chunks) if val.strip()]
    if not non_empty:
        return [""] * len(chunks)
        
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
            final = [""] * len(chunks)
            for idx, text in zip(indices, translated):
                final[idx] = text.strip()
            return final
    except Exception:
        pass

    # Fallback execution
    final = [""] * len(chunks)
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
