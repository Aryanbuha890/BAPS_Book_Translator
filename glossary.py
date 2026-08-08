import os
import re
import csv
import sqlite3

GLOSSARY_CSV_PATH = "glossary.csv"

def load_glossary(db_helper=None) -> dict[str, str]:
    """
    Loads glossary mappings from glossary.csv (startup) and the SQLite DB (per-book overrides).
    Returns a unified dict: {original_term: translated_term}
    """
    glossary = {}

    if os.path.exists(GLOSSARY_CSV_PATH):
        try:
            with open(GLOSSARY_CSV_PATH, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        src, tgt = row[0].strip(), row[1].strip()
                        if src and tgt:
                            glossary[src] = tgt
        except Exception as e:
            print(f"Warning: Failed to read {GLOSSARY_CSV_PATH}: {e}")

    if db_helper:
        try:
            db_glossary = db_helper.get_glossary()
            glossary.update(db_glossary)
        except Exception as e:
            print(f"Warning: Failed to load glossary from database: {e}")

    return glossary

def swap_in(text: str, glossary: dict[str, str]) -> tuple[str, dict[int, str]]:
    """
    Replaces glossary terms in text with unique 5-digit placeholders (80001–89999).
    One placeholder per glossary term (not per match variant) to avoid collisions.
    Uses re.sub for substitution so word-boundary logic works uniformly.

    Returns: (swapped_text, placeholder_map)
    placeholder_map: {placeholder_number: target_bengali_term}
    """
    if not glossary:
        return text, {}

    placeholder_map = {}
    counter = 80000
    swapped_text = text

    # Process longer terms first to avoid shorter sub-phrases stealing the match
    sorted_terms = sorted(glossary.keys(), key=len, reverse=True)

    # Only protect multi-word proper nouns or long single terms.
    # Single short words (< 5 chars) like 'હરિ', 'સેવા', 'સભા' are common enough
    # that the model handles them correctly. Replacing them removes context the model
    # needs to understand sentence structure.
    MIN_TERM_LENGTH = 5

    # Count how many terms would be replaced
    hit_terms = [t for t in sorted_terms if t in text and len(t) >= MIN_TERM_LENGTH]

    # Cap at 4 replacements per chunk to prevent placeholder flooding
    MAX_REPLACEMENTS = 4
    active_terms = hit_terms[:MAX_REPLACEMENTS]

    for term in active_terms:
        if term not in swapped_text:
            continue
        counter += 1
        placeholder_map[counter] = glossary[term]
        swapped_text = swapped_text.replace(term, f" {counter} ")

    swapped_text = re.sub(r'\s{2,}', ' ', swapped_text).strip()
    return swapped_text, placeholder_map

def swap_out(text: str, placeholder_map: dict[int, str]) -> str:
    """
    Restores 5-digit placeholders back to Bengali target terms.
    Handles Bengali/Devanagari/Gujarati digit variants that the model may output.
    """
    if not placeholder_map:
        return text

    normalized_text = text

    # Convert Bengali digits → ASCII
    mapping_ben = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    normalized_text = normalized_text.translate(mapping_ben)

    # Convert Devanagari digits + Gujarati digits (including ૦ zero) → ASCII
    mapping_ind = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")
    normalized_text = normalized_text.translate(mapping_ind)

    # Restore placeholders the model may have split with commas/spaces.
    # Handles: 80,001 | 8,0001 | 8 0001 | 80 001 — all variants of 5-digit 8XXXX numbers
    normalized_text = re.sub(r'\b(8[0-9])[,\s]+([0-9]{3})\b', r'\1\2', normalized_text)  # 80,001
    normalized_text = re.sub(r'\b8[,\s]+([0-9]{4})\b', r'8\1', normalized_text)            # 8,0001

    # First pass: regex word-boundary replacement
    pattern = re.compile(r'\b(8\d{4})\b')
    def replace_match(match):
        num = int(match.group(1))
        return placeholder_map.get(num, match.group(0))

    restored_text = pattern.sub(replace_match, normalized_text)

    # Second pass: direct substring replace for any placeholders joined with suffixes
    for placeholder, target_term in sorted(placeholder_map.items(), reverse=True):
        restored_text = restored_text.replace(str(placeholder), target_term)

    restored_text = re.sub(r'\s{2,}', ' ', restored_text)
    return restored_text.strip()
