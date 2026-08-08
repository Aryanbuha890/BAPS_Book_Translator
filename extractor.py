import os
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def detect_language(text: str) -> str:
    sample = text[:20000]
    if not sample.strip():
        return "unknown"
    guj_count = len(re.findall(r'[઀-૿]', sample))
    hin_count = len(re.findall(r'[ऀ-ॿ]', sample))
    if guj_count > 50:
        return "guj_Gujr"
    if hin_count > 50:
        return "hin_Deva"
    return "eng_Latn"

def extract_text_from_file(file_path: str, use_ocr_fallback: bool = True) -> tuple:
    """
    Auto-detects file extension, extracts content, and returns:
      (list_of_chapters, detected_lang_tag, empty_page_count)

    Chapters: [(chapter_title, chapter_text), ...]
    empty_page_count: number of pages/sections that had no extractable text (image-only scans).

    use_ocr_fallback: if True and pytesseract is available, runs Tesseract on empty PDF pages
                      to recover text from scanned images.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        chapters, empty_page_count = _extract_pdf(file_path, use_ocr_fallback=use_ocr_fallback)
    elif ext == '.epub':
        chapters = _extract_epub(file_path)
        empty_page_count = 0
    elif ext == '.txt':
        chapters = _extract_txt(file_path)
        empty_page_count = 0
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF, EPUB, and TXT are supported.")

    # Sample from start, middle and end to reduce cover-page bias
    sample_chunks = []
    if chapters:
        sample_chunks.append(chapters[0][1])
        if len(chapters) > 2:
            sample_chunks.append(chapters[len(chapters) // 2][1])
        if len(chapters) > 1 and len(chapters) != len(chapters) // 2:
            sample_chunks.append(chapters[-1][1])
    sample_text = " ".join(sample_chunks)
    detected_lang = detect_language(sample_text)

    return chapters, detected_lang, empty_page_count

def _try_ocr_page(page) -> str:
    """
    Renders a PyMuPDF page to an image and runs Tesseract OCR for Gujarati.
    Returns the OCR'd text, or "" if pytesseract/Pillow is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang='guj')
        return text.strip()
    except Exception:
        return ""

def _extract_pdf(file_path: str, use_ocr_fallback: bool = True) -> tuple:
    """
    Extracts PDF text page by page. For pages with no text layer (scanned images),
    optionally falls back to Tesseract OCR.
    Returns (chapters, empty_page_count).
    """
    doc = fitz.open(file_path)
    chapters = []
    empty_page_count = 0

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
                    page_lines.append(f"## {text}")
                else:
                    page_lines.append(text)
            page_lines.append("")

        chapter_text = "\n".join(page_lines).strip("\n")

        # OCR fallback for image-only pages
        if not chapter_text and use_ocr_fallback:
            chapter_text = _try_ocr_page(page)
            if not chapter_text:
                empty_page_count += 1
        elif not chapter_text:
            empty_page_count += 1

        if chapter_text:
            chapters.append((f"Page {page_num + 1}", chapter_text))

    doc.close()
    return chapters, empty_page_count

def _extract_epub(file_path: str) -> list:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError(
            "The 'ebooklib' package is required to process EPUB files. "
            "Please run 'pip install ebooklib' in your environment."
        )

    chapters = []
    book = epub.read_epub(file_path)

    spine_items = []
    for item_ref in book.spine:
        item_id = item_ref[0] if isinstance(item_ref, tuple) else item_ref
        item = book.get_item_with_id(item_id)
        if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
            spine_items.append(item)

    for idx, item in enumerate(spine_items):
        html_content = item.get_content()
        soup = BeautifulSoup(html_content, 'html.parser')

        heading = soup.find(['h1', 'h2', 'h3', 'h4'])
        title = heading.get_text().strip() if heading else f"Chapter {idx + 1}"

        paragraphs = []
        for block in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = block.get_text().strip()
            if text:
                paragraphs.append(text)

        if not paragraphs:
            for block in soup.find_all('div'):
                if not block.find('div') and not block.find('p'):
                    text = block.get_text().strip()
                    if text:
                        paragraphs.append(text)

        if not paragraphs:
            chapter_text = soup.get_text(separator="\n\n").strip()
        else:
            chapter_text = "\n\n".join(paragraphs)

        if chapter_text.strip():
            chapters.append((title, chapter_text))

    return chapters

def _extract_txt(file_path: str) -> list:
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
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

    title = os.path.basename(file_path)
    return [(title, text)]

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts Gujarati text from an image file using Tesseract OCR.
    Requires: pip install pytesseract Pillow
    System: tesseract binary with guj.traineddata installed.
    Returns the extracted text string.
    """
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # Use Gujarati + English (eng handles numbers/punctuation better)
        text = pytesseract.image_to_string(img, lang='guj+eng')
        return text.strip()
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for image OCR. "
            "Run: pip install pytesseract Pillow\n"
            "Also install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki"
        )
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")
