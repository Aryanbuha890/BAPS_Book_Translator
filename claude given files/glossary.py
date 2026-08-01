"""
glossary.py

Protects proper nouns and fixed terms (deity names, guru names, place
names, Sanskrit terms) from being mistranslated or garbled by the MT
model. Works by swapping each glossary term for a placeholder token BEFORE
translation (so the model passes it through untouched) and swapping the
correct Bengali form back in AFTER translation.

Glossary CSV format (UTF-8, no header row needed but recommended):
    source_term,bengali_term
    Bhagwan Swaminarayan,ভগবান স্বামীনারায়ণ
    Pramukh Swami Maharaj,প্রমুখ স্বামী মহারাজ
    Akshardham,অক্ষরধাম
"""

import csv
import re
from pathlib import Path


class Glossary:
    def __init__(self, csv_path: str):
        self.terms: dict[str, str] = {}
        path = Path(csv_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0].strip():
                        self.terms[row[0].strip()] = row[1].strip()
        # Longest terms first, so "Pramukh Swami Maharaj" matches before
        # a shorter overlapping term like "Swami" would.
        self._sorted_terms = sorted(self.terms.keys(), key=len, reverse=True)

    def protect(self, text: str) -> tuple[str, dict[int, str]]:
        """
        Replace known terms in `text` with unique 5-digit numbers (80001-89999).
        Returns (protected_text, placeholder_map) where placeholder_map
        maps each number to its correct Bengali translation.
        """
        placeholder_map: dict[int, str] = {}
        protected = text
        counter = 80000
        for idx, term in enumerate(self._sorted_terms):
            if term in protected:
                counter += 1
                placeholder_map[counter] = self.terms[term]
                protected = re.sub(re.escape(term), f" {counter} ", protected)
        protected = re.sub(r'\s{2,}', ' ', protected).strip()
        return protected, placeholder_map

    def restore(self, translated_text: str, placeholder_map: dict[int, str]) -> str:
        """Swap placeholders back to their correct Bengali form after translation."""
        if not placeholder_map:
            return translated_text
            
        normalized_text = translated_text
        # Convert Bengali digits to English
        mapping_ben = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
        normalized_text = normalized_text.translate(mapping_ben)
        # Convert Devanagari / Gujarati digits to English
        mapping_ind = str.maketrans("०१२३४५६७८९૧૨૩૪૫૬૭૮૯", "0123456789123456789")
        normalized_text = normalized_text.translate(mapping_ind)
        
        # Remove commas and spaces that the model often inserts into 5-digit placeholders (e.g. '8, 0004' -> '80004')
        normalized_text = re.sub(r'8\s*,?\s*(\d{4})', r'8\1', normalized_text)
        
        restored = normalized_text
        
        # First, match using regex with word boundaries
        pattern = re.compile(r'\b(8\d{4})\b')
        def replace_match(match):
            num = int(match.group(1))
            return placeholder_map.get(num, match.group(0))
            
        restored = pattern.sub(replace_match, restored)
        
        # Second, do a direct substring replace for any remaining placeholders
        for placeholder, target_term in sorted(placeholder_map.items(), reverse=True):
            restored = restored.replace(str(placeholder), target_term)
            
        restored = re.sub(r'\s{2,}', ' ', restored)
        return restored.strip()

    def apply_to_chunks(self, chunks: list[str]) -> tuple[list[str], list[dict]]:
        """Protect a whole list of chunks. Returns (protected_chunks, maps)."""
        protected_chunks = []
        maps = []
        for chunk in chunks:
            protected, pmap = self.protect(chunk)
            protected_chunks.append(protected)
            maps.append(pmap)
        return protected_chunks, maps

    def restore_chunks(self, translated_chunks: list[str], maps: list[dict]) -> list[str]:
        """Restore glossary terms across a translated chunk list."""
        return [
            self.restore(text, pmap) for text, pmap in zip(translated_chunks, maps)
        ]
