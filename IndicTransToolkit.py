# Pure-Python fallback for IndicTransToolkit using script transliteration
import sys
try:
    from indic_transliteration import sanscript
except ImportError:
    sanscript = None

class IndicProcessor:
    def __init__(self, inference_stage="model"):
        self.inference_stage = inference_stage
        try:
            from sacremoses import MosesPunctNormalizer
            self.punc_normalizer = MosesPunctNormalizer()
        except ImportError:
            self.punc_normalizer = None

    def _get_sanscript_scheme(self, lang_tag: str) -> str:
        """
        Maps a Hugging Face language tag (e.g., 'guj_Gujr') to indic-transliteration scheme names.
        """
        parts = lang_tag.split("_")
        if len(parts) > 1:
            script_code = parts[1]
            mapping = {
                "Gujr": "gujarati",
                "Beng": "bengali",
                "Deva": "devanagari",
                "Knda": "kannada",
                "Mlym": "malayalam",
                "Taml": "tamil",
                "Telu": "telugu",
                "Guru": "gurmukhi",
                "Orya": "oriya",
            }
            return mapping.get(script_code, "devanagari")
        return "devanagari"

    def preprocess_batch(self, batch: list[str], src_lang: str, tgt_lang: str, **kwargs) -> list[str]:
        """
        Transliterates source sentences to Devanagari for Script Unification, 
        normalizes punctuation, and prepends the tokenizer control tags.
        """
        preprocessed = []
        src_scheme = self._get_sanscript_scheme(src_lang)
        
        for text in batch:
            # Normalize punctuation if sacremoses is available
            if self.punc_normalizer:
                text = self.punc_normalizer.normalize(text)
            
            # Script Unification: Convert source script to Devanagari
            if sanscript and src_scheme != "devanagari":
                text = sanscript.transliterate(text, src_scheme, sanscript.DEVANAGARI)
            
            # Format expected by tokenizer: "src_lang tgt_lang sentence"
            preprocessed.append(f"{src_lang} {tgt_lang} {text}")
            
        return preprocessed
        
    def postprocess_batch(self, batch: list[str], lang: str, **kwargs) -> list[str]:
        """
        Transliterates the model's unified Devanagari output to the target language script.
        """
        cleaned = []
        tgt_scheme = self._get_sanscript_scheme(lang)
        
        for text in batch:
            # Clean up tag indicators if present
            text = text.replace(lang, "").strip()
            
            # Script Unification: Convert from Devanagari back to target script (e.g. Bengali)
            if sanscript and tgt_scheme != "devanagari":
                text = sanscript.transliterate(text, sanscript.DEVANAGARI, tgt_scheme)
                
            cleaned.append(text)
            
        return cleaned
