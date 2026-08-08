"""
Image translation support for BAPS Book Translator.

Path A — Cloud (Claude Vision): highest accuracy, sends image data to Anthropic.
Path B — Local (Tesseract OCR): private, data stays on machine; requires tesseract + guj traineddata.

After text extraction the result is fed into the normal chunker → translator pipeline,
so all existing glossary, TM, and local-model machinery applies.
"""

import base64
import json
import os


def translate_image_claude(image_path: str, api_key: str) -> dict:
    """
    Uses Claude Vision to read Gujarati text from an image and translate to Bengali in one shot.
    Returns {"extracted_gujarati": str, "bengali_translation": str}
    Requires: pip install anthropic
    """
    import anthropic

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    media_type = media_type_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Read all Gujarati text visible in this image exactly as written. "
                        "Then translate it to Bengali, preserving BAPS Swaminarayan spiritual vocabulary "
                        "(Vachanamrut → বচনামৃত, Maharaj → মহারাজ, Swaminarayan → স্বামীনারায়ণ, etc.). "
                        "Return ONLY a JSON object with exactly two keys: "
                        "'extracted_gujarati' (the raw Gujarati text) and "
                        "'bengali_translation' (the Bengali translation). "
                        "No markdown, no extra text."
                    ),
                },
            ],
        }],
    )
    return json.loads(response.content[0].text)


def extract_gujarati_tesseract(image_path: str) -> str:
    """
    Extracts Gujarati text from an image using Tesseract OCR.
    Requires: pip install pytesseract Pillow
    System: tesseract binary + guj.traineddata (see README for install instructions).
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for local image OCR.\n"
            "Run: pip install pytesseract Pillow\n"
            "Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Download guj.traineddata from: https://github.com/tesseract-ocr/tessdata"
        )

    img = Image.open(image_path)

    # Pre-process: convert to greyscale to improve OCR accuracy
    img = img.convert("L")

    # Use guj+eng: Gujarati for the main text, eng handles page numbers/punctuation
    text = pytesseract.image_to_string(img, lang="guj+eng")
    return text.strip()
