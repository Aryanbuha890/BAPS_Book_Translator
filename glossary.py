import os
import re
import csv
import sqlite3

GLOSSARY_CSV_PATH = "glossary.csv"

def load_glossary(db_helper=None) -> dict[str, str]:
    """
    Loads glossary mappings. Combines mappings stored in:
    1. The active SQLite database (via db_helper).
    2. A local 'glossary.csv' file if present in the project directory.
    
    Returns a unified dict: {original_term: translated_term}
    """
    glossary = {}
    
    # 1. Load from local CSV file if exists
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
            
    # 2. Load from SQLite Database (overwrites CSV values if conflicts exist)
    if db_helper:
        try:
            db_glossary = db_helper.get_glossary()
            glossary.update(db_glossary)
        except Exception as e:
            print(f"Warning: Failed to load glossary from database: {e}")
            
    return glossary

def swap_in(text: str, glossary: dict[str, str]) -> tuple[str, dict[int, str]]:
    """
    Scans the text for glossary terms, replaces them with unique 5-digit numbers
    in the range 80000-89999, and stores the mapped target terms.
    
    Returns a tuple: (swapped_text, placeholder_map)
    placeholder_map maps the 5-digit number to the target translated term.
    """
    if not glossary:
        return text, {}
        
    placeholder_map = {}
    counter = 80000
    swapped_text = text
    
    # Sort terms by length in descending order to match longer phrases before shorter sub-phrases
    sorted_terms = sorted(glossary.keys(), key=len, reverse=True)
    
    for term in sorted_terms:
        # Match word boundaries or exact characters
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = pattern.findall(swapped_text)
        if matches:
            for match in set(matches):
                counter += 1
                placeholder_map[counter] = glossary[term]
                # Replace exact matches with placeholder number
                swapped_text = swapped_text.replace(match, f" {counter} ")
                
    # Normalize double spaces that may be introduced by padding spaces around placeholders
    swapped_text = re.sub(r'\s{2,}', ' ', swapped_text).strip()
    return swapped_text, placeholder_map

def swap_out(text: str, placeholder_map: dict[int, str]) -> str:
    """
    Scans translated text for 5-digit placeholders (converting any Bengali/Devanagari
    digits back to English) and restores target glossary terms.
    """
    if not placeholder_map:
        return text
        
    normalized_text = text
    # Convert Bengali digits to English digits
    mapping_ben = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    normalized_text = normalized_text.translate(mapping_ben)
    # Convert Devanagari / Gujarati digits to English
    mapping_ind = str.maketrans("०१२३४५६७८९૧૨૩૪૫૬૭૮૯", "0123456789123456789")
    normalized_text = normalized_text.translate(mapping_ind)
    
    # Remove commas and spaces that the model often inserts into 5-digit placeholders (e.g. '8, 0004' -> '80004')
    normalized_text = re.sub(r'8\s*,?\s*(\d{4})', r'8\1', normalized_text)
    
    restored_text = normalized_text
    
    # First, match using regex with word boundaries to avoid corruption
    pattern = re.compile(r'\b(8\d{4})\b')
    def replace_match(match):
        num = int(match.group(1))
        return placeholder_map.get(num, match.group(0))
        
    restored_text = pattern.sub(replace_match, restored_text)
    
    # Second, do a direct substring replace for any remaining placeholders (e.g. joined with suffixes)
    for placeholder, target_term in sorted(placeholder_map.items(), reverse=True):
        restored_text = restored_text.replace(str(placeholder), target_term)
        
    # Clean up double spaces introduced by spacer padding
    restored_text = re.sub(r'\s{2,}', ' ', restored_text)
    return restored_text.strip()
