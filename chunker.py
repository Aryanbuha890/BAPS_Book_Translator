import re

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
    - Preserves paragraph boundaries as empty string elements: ""
    - Splits text on sentence boundaries: . ? ! । (danda) and ॥ (double danda)
    - Limits each chunk to 400 characters (splits long sentences at word boundaries)
    """
    # 1. Normalize line wraps and spaces
    normalized = normalize_text_spacing(text)
    
    # 2. Process paragraph by paragraph
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
            if not sentence:
                continue
                
            if len(sentence) <= 400:
                chunks.append(sentence)
            else:
                chunks.extend(break_long_sentence(sentence, 400))
                
        # Append empty string as paragraph break marker, except after the last paragraph
        if idx < len(paragraphs) - 1:
            chunks.append("")
            
    return chunks
