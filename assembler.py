import os
import urllib.request
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from ebooklib import epub
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

FONT_FILENAME = "NotoSansBengali-Regular.ttf"
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/NotoSansBengali-Regular.ttf"

def download_bengali_font():
    """
    Downloads Noto Sans Bengali font if it doesn't already exist locally.
    """
    if not os.path.exists(FONT_FILENAME):
        try:
            # Request with standard browser User-Agent to avoid blocks
            req = urllib.request.Request(
                FONT_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(FONT_FILENAME, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            # Fallback URL if main goes down
            fallback_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf"
            try:
                req = urllib.request.Request(
                    fallback_url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(FONT_FILENAME, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as ex:
                print(f"Failed to download Bengali font: {ex}")

def register_font():
    """
    Registers the TrueType Bengali font with ReportLab.
    """
    download_bengali_font()
    if os.path.exists(FONT_FILENAME):
        try:
            pdfmetrics.registerFont(TTFont('NotoSansBengali', FONT_FILENAME))
            return True
        except Exception as e:
            print(f"Error registering font: {e}")
    return False

def reconstruct_paragraphs(chunks: list[str]) -> list[str]:
    """
    Groups sentence chunks separated by empty strings back into paragraphs.
    """
    paragraphs = []
    current_para = []
    for chunk in chunks:
        if chunk == "":
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(chunk)
    if current_para:
        paragraphs.append(" ".join(current_para))
    return paragraphs

def assemble_txt(chapters_data: list[tuple[str, list[str]]], output_path: str) -> str:
    """
    Assembles chapters into a standard, double-spaced text file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for title, chunks in chapters_data:
            f.write(f"\n\n=== {title} ===\n\n")
            paragraphs = reconstruct_paragraphs(chunks)
            f.write("\n\n".join(paragraphs))
            f.write("\n")
    return output_path

def assemble_pdf(chapters_data: list[tuple[str, list[str]]], output_path: str) -> str:
    """
    Assembles chapters into a beautiful, formatted PDF using ReportLab.
    """
    has_font = register_font()
    font_name = 'NotoSansBengali' if has_font else 'Helvetica'
    
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'BengaliTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        leading=22,
        spaceAfter=15,
        textColor='#1a365d', # Deep Navy Blue
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BengaliBody',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=11,
        leading=17,
        spaceAfter=10,
        textColor='#2d3748' # Slate Grey
    )
    
    story = []
    
    # Document main title
    story.append(Paragraph("বই অনুবাদ (অনূদিত)", title_style))
    story.append(Spacer(1, 20))
    
    for idx, (title, chunks) in enumerate(chapters_data):
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 10))
        
        paragraphs = reconstruct_paragraphs(chunks)
        for para_text in paragraphs:
            if para_text.strip():
                story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 15))
        
    doc.build(story)
    return output_path

def assemble_epub(chapters_data: list[tuple[str, list[str]]], output_path: str) -> str:
    """
    Assembles chapters into a valid EPUB document using ebooklib.
    """
    book = epub.EpubBook()
    book.set_identifier('baps_translator_epub_id_1')
    book.set_title('Translated Bengali Book')
    book.set_language('bn')
    book.add_author('BAPS Book Translator')
    
    spine_items = []
    toc = []
    
    for idx, (title, chunks) in enumerate(chapters_data):
        paragraphs = reconstruct_paragraphs(chunks)
        para_html = "".join([f"<p>{p}</p>" for p in paragraphs if p.strip()])
        
        # Build HTML page
        html_content = f"""
        <?xml version="1.0" encoding="utf-8"?>
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml">
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: sans-serif; line-height: 1.6; margin: 5%; }}
                h1 {{ color: #1a365d; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }}
                p {{ margin-bottom: 1em; text-align: justify; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {para_html}
        </body>
        </html>
        """
        
        chapter_item = epub.EpubHtml(
            title=title, 
            file_name=f'chapter_{idx+1}.xhtml', 
            lang='bn'
        )
        chapter_item.content = html_content
        book.add_item(chapter_item)
        spine_items.append(chapter_item)
        toc.append(chapter_item)
        
    book.spine = ['nav'] + spine_items
    book.toc = tuple(toc)
    
    # Required navigation documents
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())
    
    epub.write_epub(output_path, book, {})
    return output_path

def assemble_output(chapters_data: list[tuple[str, list[str]]], output_dir: str, base_filename: str, format_type: str) -> str:
    """
    Top-level output assembly router. Supports 'txt', 'pdf', 'epub'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type == 'txt':
        out_path = os.path.join(output_dir, f"{base_filename}_translated.txt")
        return assemble_txt(chapters_data, out_path)
    elif format_type == 'pdf':
        out_path = os.path.join(output_dir, f"{base_filename}_translated.pdf")
        return assemble_pdf(chapters_data, out_path)
    elif format_type == 'epub':
        out_path = os.path.join(output_dir, f"{base_filename}_translated.epub")
        return assemble_epub(chapters_data, out_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
