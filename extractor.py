import os
import re
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings

# Suppress ebooklib third party warnings about UserWarning: My/File/Path is not a valid EPUB
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def detect_language(text: str) -> str:
    """
    Detects language based on character count in Unicode ranges.
    Returns: 'guj_Gujr' (Gujarati), 'hin_Deva' (Hindi), or 'eng_Latn' (English/Default)
    """
    # Sample first 10,000 characters to keep it fast
    sample = text[:10000]
    
    guj_count = len(re.findall(r'[\u0a80-\u0aff]', sample))
    hin_count = len(re.findall(r'[\u0900-\u097f]', sample))
    eng_count = len(re.findall(r'[a-zA-Z]', sample))
    
    counts = {
        "guj_Gujr": guj_count,
        "hin_Deva": hin_count,
        "eng_Latn": eng_count
    }
    
    max_lang = max(counts, key=counts.get)
    if counts[max_lang] == 0:
        return "eng_Latn"
    return max_lang

def extract_text_from_file(file_path: str) -> tuple[list[tuple[str, str]], str]:
    """
    Auto-detects file extension, extracts content page-by-page or chapter-by-chapter,
    and automatically detects the source document language.
    
    Returns a tuple: (list_of_chapters, detected_lang_tag)
    Chapters is a list of tuples: [(chapter_title, chapter_text), ...]
    detected_lang_tag is one of: 'guj_Gujr', 'hin_Deva', 'eng_Latn'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        chapters = _extract_pdf(file_path)
    elif ext == '.epub':
        chapters = _extract_epub(file_path)
    elif ext == '.txt':
        chapters = _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF, EPUB, and TXT are supported.")
        
    # Concatenate sample text to detect document language
    sample_text = " ".join([ch_text for _, ch_text in chapters[:3]])
    detected_lang = detect_language(sample_text)
    
    return chapters, detected_lang

def _extract_pdf(file_path: str) -> list[tuple[str, str]]:
    """
    Extract PDF text page by page. Each page becomes one "chapter" unit
    for simplicity; headings within a page are detected by relative font
    size and prefixed on their own line so assembler.py can style them.
    """
    doc = fitz.open(file_path)
    chapters = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception:
            blocks = []
            
        page_lines = []
        sizes = []
        for b in blocks:
            for l in b.get("lines", []):
                for span in l.get("spans", []):
                    sizes.append(span["size"])
                    
        body_size = max(set(sizes), key=sizes.count) if sizes else 10

        for block in blocks:
            # Only process text blocks, ignore image blocks (type 1)
            if block.get("type", 0) != 0:
                continue
                
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                avg_size = sum(s["size"] for s in spans) / len(spans)
                if avg_size > body_size * 1.15:
                    page_lines.append(f"## {text}")  # heading marker
                else:
                    page_lines.append(text)
            page_lines.append("")  # paragraph break after each block

        chapter_text = "\n".join(page_lines).strip("\n")
        if chapter_text:
            chapters.append((f"Page {page_num + 1}", chapter_text))

    doc.close()
    return chapters

def _extract_epub(file_path: str) -> list[tuple[str, str]]:
    """
    Extracts text from EPUB chapters following spine reading order.
    """
    chapters = []
    book = epub.read_epub(file_path)
    
    # Extract items in spine order
    spine_items = []
    for item_ref in book.spine:
        item_id = item_ref[0] if isinstance(item_ref, tuple) else item_ref
        item = book.get_item_with_id(item_id)
        if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
            spine_items.append(item)

    for idx, item in enumerate(spine_items):
        html_content = item.get_content()
        soup = BeautifulSoup(html_content, 'html.parser')

        # Try to find a heading for chapter title
        heading = soup.find(['h1', 'h2', 'h3', 'h4'])
        title = heading.get_text().strip() if heading else f"Chapter {idx + 1}"

        # Extract paragraphs & headings
        paragraphs = []
        for block in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = block.get_text().strip()
            if text:
                paragraphs.append(text)

        # Fallback to div tags if no semantic p/headings exist
        if not paragraphs:
            for block in soup.find_all('div'):
                # Avoid inner duplicates by selecting divs that don't have block children
                if not block.find('div') and not block.find('p'):
                    text = block.get_text().strip()
                    if text:
                        paragraphs.append(text)

        # Final fallback to raw text block
        if not paragraphs:
            chapter_text = soup.get_text(separator="\n\n").strip()
        else:
            chapter_text = "\n\n".join(paragraphs)

        if chapter_text.strip():
            chapters.append((title, chapter_text))
            
    return chapters

def _extract_txt(file_path: str) -> list[tuple[str, str]]:
    """
    Reads plain text files using standard encodings with fallback.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    text = ""
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='replace') as f:
                text = f.read()
            break
        except Exception:
            continue
            
    if not text:
        # Last resort fallback read
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            
    # Treat entire txt document as a single unit
    title = os.path.basename(file_path)
    return [(title, text)]
