"""
chunker.py

Splits raw extracted text into translation-ready chunks. IndicTrans2
performs best on sentence-length input, not whole paragraphs.
"""

import re

import config

# Sentence-ending punctuation: standard English + Devanagari/Gujarati danda (।)
# and double danda (॥), which end verses/shlokas.
_SENTENCE_END = re.compile(r"([.!?।॥])")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on recognized sentence-ending punctuation."""
    if not text.strip():
        return []

    parts = _SENTENCE_END.split(text)
    sentences = []
    buffer = ""
    for part in parts:
        buffer += part
        if _SENTENCE_END.fullmatch(part):
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return [s for s in sentences if s]


def _break_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break an overlong sentence at word boundaries, never mid-word."""
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split(" ")
    chunks, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = None) -> list[str]:
    """
    Convert a block of text into a list of translation-ready chunks.
    Blank lines in the source (paragraph breaks) are preserved as empty
    string entries so assembler.py can reproduce paragraph spacing.
    """
    max_chars = max_chars or config.MAX_CHUNK_CHARS
    chunks: list[str] = []

    for line in text.split("\n"):
        if not line.strip():
            chunks.append("")  # paragraph-break marker
            continue
        for sentence in split_into_sentences(line):
            chunks.extend(_break_long_sentence(sentence, max_chars))

    return chunks
