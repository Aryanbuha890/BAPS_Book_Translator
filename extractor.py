import os
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings

# Suppress ebooklib third party warnings about UserWarning: My/File/Path is not a valid EPUB
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_text_from_file(file_path: str) -> list[tuple[str, str]]:
    """
    Auto-detects file extension and extracts content.
    Returns a list of tuples: [(chapter_title, chapter_text), ...]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return _extract_pdf(file_path)
    elif ext == '.epub':
        return _extract_epub(file_path)
    elif ext == '.txt':
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF, EPUB, and TXT are supported.")

def _extract_pdf(file_path: str) -> list[tuple[str, str]]:
    """
    Extracts text from PDF page by page.
    Treats each page as a chapter block.
    """
    chapters = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            chapters.append((f"Page {page_num + 1}", text))
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
