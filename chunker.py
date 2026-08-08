import re

# Word-level OCR corrections applied BEFORE Unicode character substitutions,
# because the Unicode substitutions can corrupt these dictionary keys if run first.
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
    # Standalone fixes
    "પ રુષોત્તિ": "પુરુષોત્તમ",
    "સાંત": "સંત",
}

def clean_gujarati_ocr_typos(text: str) -> str:
    """
    Cleans OCR typos in Gujarati text.
    Order: ellipsis/repeat cleanup → word-level dict → Unicode character substitutions.
    The word-level dict must run before Unicode substitutions to avoid corrupting keys.
    """
    # 1. Remove ellipsis repeats and equals signs
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'={2,}', '', text)

    # 2. Word-level replacements (sorted by length descending to prevent substring collisions)
    sorted_replacements = sorted(GUJARATI_OCR_REPLACEMENTS.keys(), key=len, reverse=True)
    for typo in sorted_replacements:
        text = text.replace(typo, GUJARATI_OCR_REPLACEMENTS[typo])

    # 3. Systematic Unicode character corrections (run AFTER word dict to avoid key corruption)
    # U+0ABF (િ) misread combinations
    text = text.replace("િી", "મી")  # િી → મી
    text = text.replace("િી", "મી")

    text = text.replace("િા", "મા")  # િા → મા
    text = text.replace("િા", "મા")

    text = text.replace("િૃ", "મૃ")  # િૃ → મૃ
    text = text.replace("િૃ", "મૃ")

    # Isolated U+0ABF at word start is misread as 'મ'
    text = re.sub(r'(^|\s)િ', r'\1મ', text)
    text = re.sub(r'(^|\s)િ', r'\1મ', text)

    text = text.replace("હમર", "હરિ")

    return text

def normalize_text_spacing(text: str) -> str:
    """
    Normalizes line breaks: preserves paragraph breaks (double newlines),
    merges single newlines (line wraps) into spaces.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)

    paragraphs = text.split('\n\n')
    normalized_paragraphs = []
    for para in paragraphs:
        cleaned = re.sub(r'\s+', ' ', para).strip()
        if cleaned:
            normalized_paragraphs.append(cleaned)

    return '\n\n'.join(normalized_paragraphs)

def break_long_sentence(text: str, max_len: int = 300) -> list[str]:
    """
    Splits a sentence longer than max_len into smaller chunks at word boundaries.
    Max 300 chars (down from 400) to stay within IndicTrans2's subword token budget.
    """
    words = text.split()
    sub_chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        word_len = len(word)
        added_len = word_len + (1 if current_chunk else 0)

        if current_len + added_len > max_len:
            if current_chunk:
                sub_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
            else:
                sub_chunks.append(word[:max_len])
                current_chunk = [word[max_len:]]
                current_len = len(current_chunk[0])
        else:
            current_chunk.append(word)
            current_len += added_len

    if current_chunk and current_chunk[0]:
        sub_chunks.append(" ".join(current_chunk))

    return sub_chunks

# Gujarati conjunctions where long sentences can be safely split.
# Only applied when the resulting sub-sentences would each be ≤ 300 chars.
# Sorted longest-first to avoid matching sub-patterns first.
_GUJARATI_CONJUNCTIONS = [
    "કારણ કે",   # because
    "એવી રીતે",  # in the same way / similarly
    "એટલા માટે", # therefore / that is why
    "ત્યારે",    # then / at that time
    "અને",       # and
    "એટલે",      # so / therefore
    "પણ",        # but / also
    "તેથી",      # therefore
    "જ્યારે",    # when
]

def _split_at_conjunctions(sentence: str, max_len: int = 300) -> list[str]:
    """
    If a sentence exceeds max_len, try to split it at Gujarati conjunctions.
    Only splits if both resulting halves are meaningful (>10 chars each).
    Returns a list of one or more sub-sentences.
    """
    if len(sentence) <= max_len:
        return [sentence]

    for conj in _GUJARATI_CONJUNCTIONS:
        # Find the conjunction closest to the midpoint — avoids very uneven splits
        mid = len(sentence) // 2
        best_pos = -1
        best_dist = len(sentence)

        idx = 0
        while True:
            pos = sentence.find(conj, idx)
            if pos == -1:
                break
            # Don't split at the very start of a sentence
            if pos > 10:
                dist = abs(pos - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos
            idx = pos + 1

        if best_pos != -1:
            left  = sentence[:best_pos].strip()
            right = sentence[best_pos:].strip()
            # Only accept the split if both halves are substantial
            if len(left) > 15 and len(right) > 15:
                # Recursively split each half if still too long
                parts = []
                for half in [left, right]:
                    parts.extend(_split_at_conjunctions(half, max_len))
                return parts

    # No good conjunction split found — fall back to word-boundary split
    return break_long_sentence(sentence, max_len)


def chunk_text(text: str) -> list[str]:
    """
    Chunks a chapter's text into translation-friendly sentence strings.
    - Cleans OCR typos first (word-level dict, then Unicode substitutions)
    - Preserves paragraph boundaries as empty string "" markers
    - Splits on sentence boundaries: . ? ! ; । ॥
    - For sentences > 300 chars, splits at Gujarati conjunctions before falling
      back to word-boundary splitting — reduces hallucination on long inputs
    """
    # 1. Clean OCR typos (word-level dict first, then Unicode)
    cleaned_text = clean_gujarati_ocr_typos(text)

    # 2. Normalize line wraps
    normalized = normalize_text_spacing(cleaned_text)

    # 3. Process paragraph by paragraph
    paragraphs = normalized.split('\n\n')
    chunks = []

    for idx, para in enumerate(paragraphs):
        if not para.strip():
            continue

        sentences = re.split(r'(?<=[.!?;।॥])\s+', para)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or re.match(r'^\d+$', sentence):
                continue

            if len(sentence) <= 300:
                chunks.append(sentence)
            else:
                # Try conjunction split before falling back to word-boundary split
                chunks.extend(_split_at_conjunctions(sentence, 300))

        if idx < len(paragraphs) - 1:
            chunks.append("")

    return chunks
