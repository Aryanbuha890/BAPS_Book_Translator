import re

# Dictionary of common Gujarati OCR typos and their correct spellings
GUJARATI_OCR_REPLACEMENTS = {
    "ગઢડા પ્રથિ": "ગઢડા પ્રથમ",
    "સાતના": "સાતમા",
    "દૃષ્ાાંત": "દૃષ્ટાંત",
    "એિાાં": "એમાં",
    "એિા": "એમાં",
    "અમનિ": "અગ્નિ",
    "અમનિનો": "અગ્નિનો",
    "લોઢ ાં": "લોઢું",
    "સ્પશશ": "સ્પર્શ",
    "રાંગ": "રંગ",
    "શ્યાિ": "શ્યામ",
    "ગરિ": "ગરમ",
    "વણશ": "વર્ણ",
    "ગ ણાતીત": "ગુણાતીત",
    "સદ્પ રુષ": "સત્પુરુષ",
    "મનવાસ": "નિવાસ",
    "મનષ્ક ળાનાંદ": "નિષ્કુળાનંદ",
    "સિૂહ": "સન્મુખ",
    "મશષ્ય": "શિષ્ય",
    "મનિાશણીપણ ાં": "નિર્માનીપણું",
    "દૃમષ્": "દ્રષ્ટિ",
    "સારાંગપ ર": "સારંગપુર",
    "અાંદર": "અંદર",
    "અાંગે": "અંગે",
    "તેિના": "તેમના",
    "એિના": "એમના",
    "સુજાણ ને": "સુજાણતે",
    "પાવિાાં": "પાવમાં",
    "રઘ ના": "રઘુના",
    "સિજણ": "સમજણ",
    "સિજ": "સમજ",
    "મરવાજ": "મહારાજ",
    "સાંપૂણશપણે": "સંપૂર્ણપણે",
    "મહાંત": "મહંત",
    "ઉલટી પલટ્ ાં": "ઉલટી પલટું",
    "પોતાન ાં": "પોતાપણું",
    "મનિાશણીપણ": "નિર્માનીપણું",
    "વાતો કરે": "વાત કરે",
    "બોલે છે": "બોલે છે",
}

def clean_gujarati_ocr_typos(text: str) -> str:
    """
    Cleans up common OCR typos in Gujarati text (like 'િ' misread as 'મ' or 'ર' and spacing issues).
    """
    # 1. Replace ellipsis/repeats (frequent cause of translation repetition loops)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'={2,}', '', text)  # remove repeating equals signs
    
    # 2. OCR spacing merges
    text = text.replace("પ રુષોત્તિ", "પુરુષોત્તમ")
    text = text.replace("સાંત", "સંત")
    
    # 3. Systematic Unicode corrections
    text = text.replace("\u0abf\u0ac0", "મી") # િી -> મી
    text = text.replace("િી", "મી")
    
    text = text.replace("\u0abf\u0abe", "મા") # િા -> મા
    text = text.replace("િા", "મા")
    
    text = text.replace("\u0abf\u0ac3", "મૃ") # િૃ -> મૃ
    text = text.replace("િૃ", "મૃ")
    
    # Prefix 'િ' U+0ABF at word boundaries is misread 'મ' U+0AAE
    text = re.sub(r'(^|\s)\u0abf', r'\1મ', text)
    text = re.sub(r'(^|\s)િ', r'\1મ', text)
    
    text = text.replace("હમર", "હરિ")
    
    # 4. Word-level replacements (sorted by length descending to prevent substring collisions)
    sorted_replacements = sorted(GUJARATI_OCR_REPLACEMENTS.keys(), key=len, reverse=True)
    for typo in sorted_replacements:
        text = text.replace(typo, GUJARATI_OCR_REPLACEMENTS[typo])
        
    return text

def normalize_text_spacing(text: str) -> str:
    """
    Normalizes text line breaks:
    - Standardizes all carriage returns to standard newlines.
    - Preserves double newlines (paragraph breaks).
    - Merges single newlines (line wraps) into spaces within a paragraph.
    """
    # Standardize carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Compress multiple consecutive newlines (3 or more) down to 2 (a clean paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split into logical paragraphs
    paragraphs = text.split('\n\n')
    normalized_paragraphs = []
    
    for para in paragraphs:
        # Inside a paragraph, replace single newlines and multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', para).strip()
        if cleaned:
            normalized_paragraphs.append(cleaned)
            
    return '\n\n'.join(normalized_paragraphs)

def break_long_sentence(text: str, max_len: int = 400) -> list[str]:
    """
    Splits a sentence longer than max_len into smaller chunks at word boundaries.
    """
    words = text.split()
    sub_chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        word_len = len(word)
        # Adding 1 for space separation between words
        added_len = word_len + (1 if current_chunk else 0)
        
        if current_len + added_len > max_len:
            if current_chunk:
                sub_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
            else:
                # Word itself exceeds max_len (fallback: force split word)
                sub_chunks.append(word[:max_len])
                current_chunk = [word[max_len:]]
                current_len = len(current_chunk[0])
        else:
            current_chunk.append(word)
            current_len += added_len
            
    if current_chunk and current_chunk[0]:
        sub_chunks.append(" ".join(current_chunk))
        
    return sub_chunks

def chunk_text(text: str) -> list[str]:
    """
    Chunks a chapter's text into small translation-friendly sentence strings.
    - Cleans OCR typos from text before chunking.
    - Preserves paragraph boundaries as empty string elements: ""
    - Splits text on sentence boundaries: . ? ! । (danda) and ॥ (double danda)
    - Limits each chunk to 400 characters (splits long sentences at word boundaries)
    """
    # 1. Clean OCR typos
    cleaned_text = clean_gujarati_ocr_typos(text)
    
    # 2. Normalize line wraps and spaces
    normalized = normalize_text_spacing(cleaned_text)
    
    # 3. Process paragraph by paragraph
    paragraphs = normalized.split('\n\n')
    chunks = []
    
    for idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
            
        # Split paragraph into sentences.
        # Uses positive lookbehind so that the punctuation marks are preserved in the sentences
        sentences = re.split(r'(?<=[.!?।॥])\s+', para)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty chunks and layout artifacts like raw page numbers (e.g. "1")
            if not sentence or re.match(r'^\d+$', sentence):
                continue
                
            if len(sentence) <= 400:
                chunks.append(sentence)
            else:
                chunks.extend(break_long_sentence(sentence, 400))
                
        # Append empty string as paragraph break marker, except after the last paragraph
        if idx < len(paragraphs) - 1:
            chunks.append("")
            
    return chunks
