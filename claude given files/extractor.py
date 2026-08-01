"""
extractor.py

Extracts text from PDF, EPUB, or TXT while preserving document structure
(chapter/section boundaries, paragraph breaks) so assembler.py can rebuild
a properly formatted Bengali output rather than one wall of text.

Returns a list of "chapters": (chapter_title, chapter_text) tuples, in
reading order. chapter_text uses "\n" for paragraph breaks, which
chunker.py interprets as paragraph markers.
"""

from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


def extract(file_path: str) -> list[tuple[str, str]]:
    """Auto-detect file type by extension and dispatch to the right extractor."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".epub":
        return _extract_epub(file_path)
    if suffix == ".txt":
        return _extract_txt(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(file_path: str) -> list[tuple[str, str]]:
    """
    Extract PDF text page by page. Each page becomes one "chapter" unit
    for simplicity; headings within a page are detected by relative font
    size and prefixed on their own line so assembler.py can style them.
    """
    doc = fitz.open(file_path)
    chapters = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        sizes = [
            span["size"]
            for b in blocks
            for l in b.get("lines", [])
            for span in l.get("spans", [])
        ]
        body_size = max(set(sizes), key=sizes.count) if sizes else 10

        for block in blocks:
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
            chapters.append((f"Page {page_num}", chapter_text))

    doc.close()
    return chapters


def _extract_epub(file_path: str) -> list[tuple[str, str]]:
    """Extract EPUB chapters in correct spine (reading) order."""
    book = epub.read_epub(file_path)
    chapters = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        title_tag = soup.find(["h1", "h2", "title"])
        title = title_tag.get_text(strip=True) if title_tag else item.get_name()

        lines = []
        for el in soup.find_all(["h1", "h2", "h3", "p"]):
            text = el.get_text(strip=True)
            if not text:
                continue
            if el.name in ("h1", "h2", "h3"):
                lines.append(f"## {text}")
            else:
                lines.append(text)
            lines.append("")  # paragraph break

        chapter_text = "\n".join(lines).strip("\n")
        if chapter_text:
            chapters.append((title, chapter_text))

    return chapters


def _extract_txt(file_path: str) -> list[tuple[str, str]]:
    """Read a plain text file as a single chapter."""
    with open(file_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [(Path(file_path).stem, text)]
