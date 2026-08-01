"""
app.py

Streamlit UI for the local, offline BAPS document translator.
Run with: streamlit run app.py
"""

import time
from pathlib import Path

import streamlit as st

import config
from extractor import extract
from chunker import chunk_text
from glossary import Glossary
from translator import translate_chunks, detect_source_lang
from progress import ProgressDB
from assembler import assemble_docx, assemble_txt


st.set_page_config(page_title="Gujarati/Hindi/English → Bengali Translator", layout="wide")
st.title("📖 BAPS Document Translator — Local & Private")
st.caption("English, Hindi, Gujarati → Bengali. Runs 100% on your machine. Nothing is uploaded anywhere.")

uploaded_file = st.file_uploader("Upload a PDF, EPUB, or TXT file", type=["pdf", "epub", "txt"])

if uploaded_file:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    saved_path = upload_dir / uploaded_file.name
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Loaded: {uploaded_file.name}")

    progress_db_path = saved_path.parent / f"{saved_path.stem}_progress.db"
    resume_available = progress_db_path.exists()

    mode = "Fresh start"
    if resume_available:
        mode = st.radio("An in-progress translation was found for this file:", ["Resume", "Start Fresh"])

    if st.button("Start Translation", type="primary"):
        with st.spinner("Reading and structuring document..."):
            chapters_raw = extract(str(saved_path))  # list[(title, text)]

        glossary = Glossary(config.GLOSSARY_CSV_PATH)
        db = ProgressDB(str(saved_path))

        # Flatten all chapters into a single ordered chunk list, remembering
        # which chapter each chunk belongs to, so resume works across the
        # whole book, not per chapter.
        if not resume_available or mode == "Start Fresh":
            flat_chunks = []
            for chapter_index, (title, text) in enumerate(chapters_raw):
                for chunk in chunk_text(text):
                    flat_chunks.append((chapter_index, chunk))
            db.seed_if_empty(flat_chunks)

        total, done = db.total_and_done()
        progress_bar = st.progress(done / total if total else 0)
        status_text = st.empty()
        preview_placeholder = st.empty()

        pending = db.get_pending_chunks()
        start_time = time.time()
        preview_samples = []

        for i, (chunk_id, original_text) in enumerate(pending):
            src_lang = detect_source_lang(original_text) if original_text.strip() else config.LANG_ENGLISH

            protected_text, pmap = glossary.protect(original_text)
            translated_list = translate_chunks([protected_text], src_lang)
            translated_text = glossary.restore(translated_list[0], pmap)

            db.save_chunk(chunk_id, translated_text)

            done += 1
            elapsed = time.time() - start_time
            rate = elapsed / max(done, 1)
            remaining = (total - done) * rate
            progress_bar.progress(min(done / total, 1.0))
            status_text.text(
                f"{done}/{total} chunks translated · ~{remaining/60:.1f} min remaining"
            )

            if len(preview_samples) < 3 and original_text.strip():
                preview_samples.append((original_text, translated_text))
                with preview_placeholder.container():
                    st.subheader("Live preview")
                    for orig, trans in preview_samples:
                        col1, col2 = st.columns(2)
                        col1.text(orig)
                        col2.text(trans)

        st.success("Translation complete!")

        # Rebuild chapters with translated text for the assembler
        all_chunks = db.get_all_chunks()  # (chunk_id, chapter_index, orig, translated, status)
        chapters_translated: dict[int, list[str]] = {}
        chapter_titles = {i: title for i, (title, _) in enumerate(chapters_raw)}
        for _, chapter_index, _, translated_text, _ in all_chunks:
            chapters_translated.setdefault(chapter_index, []).append(translated_text or "")

        chapters_for_output = [
            (chapter_titles.get(idx, f"Chapter {idx+1}"), chunks)
            for idx, chunks in sorted(chapters_translated.items())
        ]

        output_dir = Path(config.OUTPUT_DIR)
        docx_path = output_dir / f"{saved_path.stem}_bengali.docx"
        txt_path = output_dir / f"{saved_path.stem}_bengali.txt"
        assemble_docx(chapters_for_output, str(docx_path))
        assemble_txt(chapters_for_output, str(txt_path))

        with open(docx_path, "rb") as f:
            st.download_button("⬇️ Download Bengali Translation (.docx)", f, file_name=docx_path.name)
        with open(txt_path, "rb") as f:
            st.download_button("⬇️ Download Bengali Translation (.txt)", f, file_name=txt_path.name)

        db.close()
