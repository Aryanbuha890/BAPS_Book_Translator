import os
import re
import json
import time
import pandas as pd
import streamlit as st
import shutil

from extractor import extract_text_from_file
from chunker import chunk_text
from progress import ProgressDB
from translator import (
    translate_chunks_local,
    translate_chunks_gemini,
    translate_chunks_claude,
    clean_translated_text,
    check_translation_memory,
    save_to_translation_memory,
    reload_translation_memory
)
from glossary import swap_in, swap_out, load_glossary
from assembler import assemble_output

# Set page config for premium look
st.set_page_config(
    page_title="BAPS Book Translator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply font family globally */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header gradient styling */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 35px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .title-container h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .title-container p {
        margin: 8px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Premium card container styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #cbd5e1;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.75px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Section containers */
    .panel-container {
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Engine warning notes */
    .privacy-tag {
        display: inline-block;
        font-size: 0.8rem;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .privacy-local {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .privacy-cloud {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading IndicTrans2 model into memory… (first run only, takes 1–2 min)")
def _preload_local_model(model_path: str):
    """
    Loads and caches the IndicTrans2 model using Streamlit's resource cache.
    Runs once per session; subsequent reruns reuse the cached model.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor
    import torch
    import translator as _tr

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    processor = IndicProcessor()

    # Inject into translator module cache so translate_chunks_local finds them
    model_name = "ai4bharat/indictrans2-indic-indic-1B"
    _tr._local_tokenizers[model_name] = tokenizer
    _tr._local_models[model_name] = model
    _tr._local_processor = processor
    _tr._local_device = device
    return device.type

# Initialize Session States
if "translating" not in st.session_state:
    st.session_state.translating = False
if "book_path" not in st.session_state:
    st.session_state.book_path = None
if "current_db" not in st.session_state:
    st.session_state.current_db = None
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "last_batch_time" not in st.session_state:
    st.session_state.last_batch_time = []
if "glossary_cache" not in st.session_state:
    st.session_state.glossary_cache = {}
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = None
if "detected_lang_label" not in st.session_state:
    st.session_state.detected_lang_label = None
if "prev_translated_chunk" not in st.session_state:
    st.session_state.prev_translated_chunk = None
if "image_mode" not in st.session_state:
    st.session_state.image_mode = False
if "image_path" not in st.session_state:
    st.session_state.image_path = None

# Ensure workspace folders exist
os.makedirs("temp", exist_ok=True)
os.makedirs("db", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ----------------- SIDEBAR: Settings -----------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/book.png", width=90)
    st.title("Settings")
    st.markdown("---")
    
    # Source Language configuration
    src_lang_name = st.selectbox(
        "Source Language",
        ["Auto-detect", "Gujarati", "Hindi", "English"],
        help="Select the language of your input document. 'Auto-detect' scans the document script directly."
    )
    
    src_lang_map = {
        "Gujarati": "guj_Gujr",
        "Hindi": "hin_Deva",
        "English": "eng_Latn"
    }
    
    # Resolve source language
    if src_lang_name == "Auto-detect":
        if st.session_state.detected_lang and st.session_state.detected_lang != "unknown":
            src_lang = st.session_state.detected_lang
            st.caption(f"✓ Detected Script: **{st.session_state.detected_lang_label}**")
        else:
            src_lang = "guj_Gujr"  # Default fallback before file load
            if st.session_state.detected_lang == "unknown":
                st.caption("⚠️ Could not detect script/language. Please select Source Language manually.")
            else:
                st.caption("Upload a document to auto-detect language.")
    else:
        src_lang = src_lang_map[src_lang_name]
        
    # Engine Selection
    engine = st.selectbox(
        "Translation Engine",
        ["Local (IndicTrans2)", "Cloud (Gemini)", "Cloud (Claude)"],
        help="Local (IndicTrans2) runs 100% offline and privately on your CPU/GPU. Cloud requires internet but may handle novel sentences differently."
    )
    
    # Check compatibility/download state for local engine and preload model
    if engine == "Local (IndicTrans2)":
        if src_lang == "eng_Latn":
            model_dir = os.path.join("models", "indictrans2-en-indic-1B")
            if not os.path.isdir(model_dir):
                st.warning("⚠️ Local English-to-Bengali translation requires the English-to-Indic model. Please run `python download_model.py` and select Option 2 or 3 to download the weights.")
        else:
            model_dir = os.path.join("models", "indictrans2-indic-indic-1B")
            if os.path.isdir(model_dir):
                # Preload model now so translation reruns don't block
                _device_used = _preload_local_model(model_dir)
                st.caption(f"✓ Model loaded on **{_device_used.upper()}**")
            else:
                st.warning("⚠️ Local Indic-to-Bengali translation requires the Indic-to-Indic model. Please run `python download_model.py` and select Option 1 or 3 to download the weights.")
    
    # Engine specific configurations & notifications
    if engine == "Local (IndicTrans2)":
        st.markdown('<span class="privacy-tag privacy-local">🛡️ 100% Offline & Private</span>', unsafe_allow_html=True)
        st.info("Uses local 1B model files cached in models/ directory. Run is fully private.")
        
        # Hugging Face Token (only needed if model is not downloaded yet)
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            help="Your token is only checked if weights are missing locally. Leave blank if download is already complete."
        )
        st.caption("[Get Hugging Face Token](https://huggingface.co/settings/tokens)")
        batch_size = st.slider("Batch Size (Lower if RAM < 8GB)", 2, 16, 4)
        api_key = hf_token
    else:
        st.markdown('<span class="privacy-tag privacy-cloud">⚠️ Sends Data Online</span>', unsafe_allow_html=True)
        batch_size = st.slider("Batch Size (Sentences per Call)", 4, 20, 8)
        
        if engine == "Cloud (Gemini)":
            api_key = st.text_input("Gemini API Key", type="password", help="Grab keys from Google AI Studio")
            st.caption("[Get Google Gemini API Key](https://aistudio.google.com/)")
        else:
            api_key = st.text_input("Claude API Key", type="password", help="Grab keys from Anthropic Console")
            st.caption("[Get Anthropic API Key](https://console.anthropic.com/)")
            
    st.markdown("---")
    st.caption("v1.4.0 | Multi-Source Local & Cloud Translation")

# ----------------- MAIN INTERFACE -----------------
st.markdown("""
<div class="title-container">
    <h1>📚 BAPS Book Translator</h1>
    <p>Premium Multi-Source to Bengali Literary Translation Suite (Local & Cloud Supported)</p>
</div>
""", unsafe_allow_html=True)

# 1. File Upload Section
uploaded_file = st.file_uploader(
    "Upload Document (PDF, EPUB, TXT) or Image (PNG, JPG)",
    type=["pdf", "epub", "txt", "png", "jpg", "jpeg"],
    help="Upload a book file or a photo/scan of a page."
)

if uploaded_file:
    # Set book path
    temp_path = os.path.join("temp", uploaded_file.name)
    file_hash = hash(uploaded_file.getvalue())
    
    if st.session_state.get("file_hash") != file_hash or st.session_state.book_path != temp_path:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        st.session_state.book_path = temp_path
        st.session_state.file_hash = file_hash
        # Load progress DB
        db_helper = ProgressDB(temp_path)
        st.session_state.current_db = db_helper
        
        # Load custom terminology
        st.session_state.glossary_cache = load_glossary(db_helper)
        
        # Handle image files separately
        image_exts = {".png", ".jpg", ".jpeg"}
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if file_ext in image_exts:
            # Image mode — show OCR + translate UI below; no DB setup here
            st.session_state.image_mode = True
            st.session_state.image_path = temp_path
            st.session_state.current_db = None
        else:
            st.session_state.image_mode = False
            # Extract and parse file
            with st.spinner("Extracting content and parsing chapters..."):
                try:
                    chapters, detected_lang, empty_page_count = extract_text_from_file(temp_path)

                    total_text_length = sum(len(text.strip()) for _, text in chapters) if chapters else 0
                    if total_text_length == 0:
                        st.error("⚠️ No text could be extracted from this document. It might be a scanned PDF or empty. Use 'Local OCR' engine or upload a text-layer PDF.")
                        st.session_state.book_path = None
                        st.session_state.current_db = None
                        st.session_state.detected_lang = None
                        st.session_state.detected_lang_label = None
                        st.session_state.file_hash = None
                    else:
                        if empty_page_count > 0:
                            st.warning(
                                f"⚠️ {empty_page_count} page(s) appear to be image-only scans with no text layer. "
                                "Those pages will be blank in the translation. Use 'Local OCR' engine or a text-layer PDF for full coverage."
                            )

                        lang_labels = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "unknown": "Unknown"}
                        st.session_state.detected_lang = detected_lang
                        st.session_state.detected_lang_label = lang_labels.get(detected_lang, "Unknown")

                        chapters_chunked = []
                        for title, text in chapters:
                            chunks = chunk_text(text)
                            chapters_chunked.append((title, chunks))

                        stats = db_helper.get_progress_stats()
                        if stats['total_chunks'] > 0:
                            db_chunks = [c['original_text'] for c in db_helper.get_all_chunks()]
                            new_chunks = [chunk for _, chunks in chapters_chunked for chunk in chunks]
                            if db_chunks != new_chunks:
                                with db_helper.conn:
                                    db_helper.conn.execute("DELETE FROM translation_progress")

                        is_new = db_helper.initialize_chunks(chapters_chunked)
                        if is_new:
                            st.toast("Success! Initialized new book translation database.", icon="✨")
                        else:
                            st.toast("Welcome back! Loaded existing translation progress.", icon="♻️")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.session_state.book_path = None
                    st.session_state.current_db = None

# ----------------- IMAGE MODE UI -----------------
if st.session_state.get("image_mode") and st.session_state.get("image_path"):
    image_path = st.session_state.image_path
    st.markdown("---")
    st.markdown("### 🖼️ Image Translation Mode")

    from PIL import Image as PILImage
    try:
        img = PILImage.open(image_path)
        st.image(img, caption="Uploaded image", use_column_width=True)
    except Exception:
        pass

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("#### Path A — Cloud Vision (Claude)")
        st.caption("Sends image to Claude. Highest accuracy — requires Claude API key.")
        cloud_key_img = st.text_input("Claude API Key (for image)", type="password", key="img_claude_key")
        if st.button("Translate Image via Claude Vision", key="btn_img_claude"):
            if not cloud_key_img:
                st.error("Enter your Claude API key above.")
            else:
                with st.spinner("Sending image to Claude Vision..."):
                    try:
                        from image_translator import translate_image_claude
                        result = translate_image_claude(image_path, cloud_key_img)
                        st.success("Done!")
                        st.text_area("Extracted Gujarati", result.get("extracted_gujarati", ""), height=200, key="img_guj_cloud")
                        st.text_area("Bengali Translation", result.get("bengali_translation", ""), height=200, key="img_ben_cloud")
                    except Exception as e:
                        st.error(f"Cloud Vision failed: {e}")

    with col_img2:
        st.markdown("#### Path B — Local OCR (Tesseract)")
        st.caption("100% offline. Requires Tesseract binary + guj.traineddata installed on this machine.")
        if st.button("Extract Text via Tesseract OCR", key="btn_img_ocr"):
            with st.spinner("Running Tesseract OCR..."):
                try:
                    from image_translator import extract_gujarati_tesseract
                    extracted = extract_gujarati_tesseract(image_path)
                    st.session_state["ocr_extracted"] = extracted
                except Exception as e:
                    st.error(f"OCR failed: {e}\n\nMake sure Tesseract is installed and guj.traineddata is present.")

        if st.session_state.get("ocr_extracted"):
            guj_text = st.text_area("Extracted Gujarati (editable — fix OCR errors before translating)",
                                     st.session_state["ocr_extracted"], height=200, key="img_guj_ocr")
            if st.button("Translate via Local Model", key="btn_img_local_translate"):
                with st.spinner("Translating with local IndicTrans2 model..."):
                    try:
                        from chunker import chunk_text
                        chunks = chunk_text(guj_text)
                        non_empty = [c for c in chunks if c.strip()]
                        if non_empty:
                            translated = translate_chunks_local(
                                non_empty, src_lang="guj_Gujr", tgt_lang="ben_Beng",
                                batch_size=4, hf_token=api_key if engine == "Local (IndicTrans2)" else None
                            )
                            st.text_area("Bengali Translation", " ".join(t for t in translated if t), height=200, key="img_ben_ocr")
                        else:
                            st.warning("No text found to translate.")
                    except Exception as e:
                        st.error(f"Translation failed: {e}")

# Main content dashboard when book is loaded
if st.session_state.current_db:
    db_helper = st.session_state.current_db
    stats = db_helper.get_progress_stats()
    st.session_state.stats = stats
    
    # Initialize workspace tabs
    tab_dash, tab_editor, tab_tm, tab_glossary = st.tabs([
        "📖 Translation Dashboard",
        "✍️ Side-by-Side Editor",
        "🧠 Translation Memory",
        "🏷️ Terminology Glossary"
    ])
    
    # ----------------- TAB 1: Translation Dashboard -----------------
    with tab_dash:
        # Row of Metrics Dashboard
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Progress</div>
                <div class="metric-value">{stats['percent_complete']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Translated Chunks</div>
                <div class="metric-value">{stats['completed_chunks']} / {stats['total_chunks']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pending Chunks</div>
                <div class="metric-value">{stats['pending_chunks']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">User Corrections</div>
                <div class="metric-value">{stats['user_edited_chunks']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Execution buttons panel
        st.markdown("### 🎮 Translation Controls")
        col_btn1, col_btn2, col_btn3, col_empty = st.columns([1.5, 1.5, 1.5, 3.5])
        
        with col_btn1:
            if stats['percent_complete'] < 100:
                btn_label = "▶️ Start Translation" if stats['completed_chunks'] == 0 else "▶️ Resume Translation"
                if st.button(btn_label, use_container_width=True, disabled=st.session_state.translating):
                    if engine == "Local (IndicTrans2)":
                        if src_lang == "eng_Latn":
                            model_dir = os.path.join("models", "indictrans2-en-indic-1B")
                            if not os.path.isdir(model_dir):
                                st.error("Local English-to-Bengali model weights not found. Please run `python download_model.py` and select Option 2 or 3 first.")
                            else:
                                st.session_state.translating = True
                                st.rerun()
                        else:
                            model_dir = os.path.join("models", "indictrans2-indic-indic-1B")
                            if not os.path.isdir(model_dir):
                                st.error("Local Indic-to-Bengali model weights not found. Please run `python download_model.py` and select Option 1 or 3 first.")
                            else:
                                st.session_state.translating = True
                                st.rerun()
                    elif engine != "Local (IndicTrans2)" and not api_key:
                        st.error("API Key is required for cloud translation engines!")
                    else:
                        st.session_state.translating = True
                        st.rerun()
            else:
                st.button("✅ Completed", disabled=True, use_container_width=True)
                
        with col_btn2:
            if st.button("⏸ Pause Translation", use_container_width=True, disabled=not st.session_state.translating):
                st.session_state.translating = False
                st.rerun()
                
        with col_btn3:
            if st.button("🔄 Reset Progress", use_container_width=True):
                st.session_state.translating = False
                st.session_state.prev_translated_chunk = None
                # Close connection first to avoid file-lock on Windows, then reopen
                db_helper.close()
                import sqlite3 as _sqlite3
                _conn = _sqlite3.connect(db_helper.db_path, check_same_thread=False)
                with _conn:
                    _conn.execute("DELETE FROM translation_progress")
                _conn.close()
                # Reopen fresh connection
                db_helper.conn = _sqlite3.connect(db_helper.db_path, check_same_thread=False)
                st.session_state.current_db = db_helper
                st.toast("Progress reset! Please re-upload the document or reload the page to load fresh chunks.", icon="🔄")
                st.rerun()
                
        st.markdown("---")
        
        # Active loop execution
        if st.session_state.translating:
            pending_chunks = db_helper.get_pending_chunks()
            
            if not pending_chunks:
                st.session_state.translating = False
                st.success("Hooray! The book translation has finished successfully.")
                st.rerun()
            else:
                # Active batch translation
                batch = pending_chunks[:batch_size]
                batch_ids = [c[0] for c in batch]
                batch_texts = [c[1] for c in batch]
                # Apply Glossary Swap-in (Placeholder system)
                swapped_texts = []
                placeholder_maps = []
                for t in batch_texts:
                    s_text, p_map = swap_in(t, st.session_state.glossary_cache)
                    swapped_texts.append(s_text)
                    placeholder_maps.append(p_map)
                
                # Set translating status in database
                for c_id in batch_ids:
                    db_helper.update_chunk_translating(c_id)
                    
                # Perform translation
                t0 = time.time()
                try:
                    if engine == "Local (IndicTrans2)":
                        translated_batch = translate_chunks_local(swapped_texts, src_lang=src_lang, tgt_lang="ben_Beng", batch_size=batch_size, hf_token=api_key)
                    elif engine == "Cloud (Gemini)":
                        translated_batch = translate_chunks_gemini(swapped_texts, api_key=api_key, src_lang=src_lang, tgt_lang="ben_Beng", glossary=st.session_state.glossary_cache, prev_context=st.session_state.prev_translated_chunk)
                    else:
                        translated_batch = translate_chunks_claude(swapped_texts, api_key=api_key, src_lang=src_lang, tgt_lang="ben_Beng", glossary=st.session_state.glossary_cache, prev_context=st.session_state.prev_translated_chunk)
                        
                    # Apply Glossary Swap-out
                    final_translated = []
                    for trans_t, p_map in zip(translated_batch, placeholder_maps):
                        r_text = swap_out(trans_t, p_map)
                        final_translated.append(r_text)
                        
                    # Save translations and track last non-empty translated chunk for context
                    for c_id, trans_txt in zip(batch_ids, final_translated):
                        db_helper.save_chunk(c_id, trans_txt)
                        if trans_txt and trans_txt.strip():
                            st.session_state.prev_translated_chunk = trans_txt.strip()

                    # Speed & ETA calculation
                    time_taken = time.time() - t0
                    st.session_state.last_batch_time.append(time_taken / len(batch))
                    if len(st.session_state.last_batch_time) > 20:
                        st.session_state.last_batch_time.pop(0)
                except Exception as ex:
                    st.error(f"Translation call failed: {ex}")
                    st.session_state.translating = False
                    st.rerun()
                    
                st.rerun()
                
        # Live status tracking indicators
        if st.session_state.translating and st.session_state.last_batch_time:
            avg_chunk_time = sum(st.session_state.last_batch_time) / len(st.session_state.last_batch_time)
            eta_sec = stats['pending_chunks'] * avg_chunk_time
            eta_min = eta_sec / 60
            
            st.markdown("### ⚡ Live Status Tracker")
            st.progress(stats['percent_complete'] / 100.0)
            
            col_st1, col_st2, col_st3 = st.columns(3)
            with col_st1:
                st.metric("Processing Speed", f"{1/avg_chunk_time:.2f} sentences/sec")
            with col_st2:
                st.metric("Estimated Time Remaining", f"{eta_min:.1f} minutes")
            with col_st3:
                st.metric("Batch Average Time", f"{avg_chunk_time:.2f}s")
                
            st.info("Translating next paragraph... Click 'Pause Translation' to halt at the next batch.")
            
        # Reassemble & Download translated assets
        st.markdown("### 📥 Reassemble & Download Book")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        # Build assembled chapters dynamically with post-assembly corrections and translation memory
        all_chunks = db_helper.get_all_chunks()
        chapters_dict = {}
        for chunk in all_chunks:
            ch_idx = chunk['chapter_index']
            ch_title = chunk['chapter_title']
            orig = chunk['original_text']
            trans = chunk['translated_text']
            
            # Post-assembly validation: Apply translation memory and cleanup
            # unless the user has manually edited this chunk
            if chunk['modified_by_user'] == 0 and trans:
                mem_match = check_translation_memory(orig)
                if mem_match:
                    trans = mem_match
                else:
                    trans = clean_translated_text(trans)
                    
            if ch_idx not in chapters_dict:
                chapters_dict[ch_idx] = (ch_title, [])
            chapters_dict[ch_idx][1].append(trans)
            
        sorted_indices = sorted(chapters_dict.keys())
        chapters_assembled = [chapters_dict[idx] for idx in sorted_indices]
        book_base_name = os.path.splitext(os.path.basename(st.session_state.book_path))[0]
        
        with col_d1:
            st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
            st.markdown("##### 📄 Text/Markdown Format")
            txt_path = os.path.join("output", f"{book_base_name}_translated.txt")
            if st.button("Generate Text File", key="gen_txt", use_container_width=True):
                with st.spinner("Assembling text file..."):
                    assemble_output(chapters_assembled, "output", book_base_name, "txt")
                    st.success("Text File Generated!")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download Text File",
                        data=f.read(),
                        file_name=os.path.basename(txt_path),
                        mime="text/plain",
                        use_container_width=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_d2:
            st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
            st.markdown("##### 📕 E-Reader EPUB Format")
            epub_path = os.path.join("output", f"{book_base_name}_translated.epub")
            if st.button("Generate EPUB File", key="gen_epub", use_container_width=True):
                with st.spinner("Assembling EPUB file..."):
                    assemble_output(chapters_assembled, "output", book_base_name, "epub")
                    st.success("EPUB File Generated!")
            if os.path.exists(epub_path):
                with open(epub_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download EPUB Book",
                        data=f.read(),
                        file_name=os.path.basename(epub_path),
                        mime="application/epub+zip",
                        use_container_width=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_d3:
            st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
            st.markdown("##### 📄 PDF Document (w/ Bengali Font)")
            pdf_path = os.path.join("output", f"{book_base_name}_translated.pdf")
            if st.button("Generate PDF Document", key="gen_pdf", use_container_width=True):
                with st.spinner("Assembling PDF document..."):
                    assemble_output(chapters_assembled, "output", book_base_name, "pdf")
                    st.success("PDF Document Compiled!")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Book",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d4:
            st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
            st.markdown("##### 📁 MS Word (.docx) Document")
            docx_path = os.path.join("output", f"{book_base_name}_translated.docx")
            if st.button("Generate Word File", key="gen_docx", use_container_width=True):
                with st.spinner("Assembling Word document..."):
                    assemble_output(chapters_assembled, "output", book_base_name, "docx")
                    st.success("Word File Generated!")
            if os.path.exists(docx_path):
                with open(docx_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Word Document",
                        data=f.read(),
                        file_name=os.path.basename(docx_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- TAB 2: Interactive Editor -----------------
    with tab_editor:
        st.subheader("📝 Side-by-Side Sentence Inspector & Editor")
        st.write(
            "Double-click any cell in the **Translated Bengali (Editable)** column to make adjustments. "
            "Changes are saved instantly. Use **Save to Memory** to permanently add verified pairs to the translation memory."
        )

        # Load all chunks for data editor
        raw_chunks = db_helper.get_all_chunks()
        if raw_chunks:
            df_editor = pd.DataFrame(raw_chunks)

            edited_df = st.data_editor(
                df_editor,
                column_config={
                    "chunk_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "chapter_index": None,
                    "chapter_title": st.column_config.TextColumn("Chapter", disabled=True, width="medium"),
                    "original_text": st.column_config.TextColumn("Original Text", disabled=True, width="large"),
                    "translated_text": st.column_config.TextColumn("Translated Bengali (Editable)", width="large"),
                    "status": st.column_config.TextColumn("Status", disabled=True),
                    "modified_by_user": None,
                },
                width="stretch",
                num_rows="fixed",
                key="sentence_editor"
            )

            # Save edited values back to database if changed
            if not edited_df.equals(df_editor):
                diff_mask = edited_df["translated_text"] != df_editor["translated_text"]
                for idx, row in edited_df[diff_mask].iterrows():
                    db_helper.mark_chunk_done(
                        chunk_id=int(row["chunk_id"]),
                        translated_text=row["translated_text"],
                        modified_by_user=1
                    )
                st.toast("Sentence changes saved to database!", icon="💾")
                st.rerun()

            # Save-to-Memory: persist user-verified pairs permanently
            st.markdown("---")
            st.markdown("#### 🧠 Save Verified Translations to Translation Memory")
            st.write(
                "Select a chunk ID to save its current original→Bengali pair permanently to `translation_memory.json`. "
                "Saved pairs will be used for exact/fuzzy matching on future books — no model needed."
            )
            user_edited_rows = edited_df[edited_df["modified_by_user"] == 1]
            if not user_edited_rows.empty:
                save_options = {
                    f"[{int(row['chunk_id'])}] {row['original_text'][:60]}…": int(row['chunk_id'])
                    for _, row in user_edited_rows.iterrows()
                    if row.get("original_text") and row.get("translated_text")
                }
                selected_label = st.selectbox("Pick a verified sentence to save:", list(save_options.keys()), key="tm_save_select")
                if st.button("✅ Save to Translation Memory", key="btn_save_tm"):
                    selected_id = save_options[selected_label]
                    selected_row = edited_df[edited_df["chunk_id"] == selected_id].iloc[0]
                    save_to_translation_memory(selected_row["original_text"], selected_row["translated_text"])
                    st.success(f"Saved to translation_memory.json: '{selected_row['original_text'][:50]}…'")
                    reload_translation_memory()
            else:
                st.info("No manually-edited sentences yet. Edit a Bengali translation above to enable this feature.")

    # ----------------- TAB 3: Translation Memory Manager -----------------
    with tab_tm:
        st.subheader("🧠 Translation Memory Manager")
        st.write(
            "The Translation Memory (TM) gives **100% accurate** translations instantly — "
            "no model needed. Every pair added here improves ALL future book translations permanently."
        )

        # Load current TM
        _tm_path = "translation_memory.json"
        with open(_tm_path, "r", encoding="utf-8") as _f:
            _tm_data = json.load(_f)
        st.metric("Current TM size", f"{len(_tm_data)} verified sentence pairs")
        st.markdown("---")

        # ── PATH 1: Batch save all corrected sentences from current book ──────
        st.markdown("### Path 1 — Save Verified Sentences from This Book")
        st.write(
            "Every sentence you corrected in the Editor tab is a verified pair. "
            "Save them all to TM in one click."
        )
        _all_chunks = db_helper.get_all_chunks()
        _corrected = [c for c in _all_chunks if c["modified_by_user"] == 1
                      and c["original_text"].strip() and c["translated_text"].strip()]
        _already_in_tm = sum(1 for c in _corrected if c["original_text"] in _tm_data)
        _new_count = len(_corrected) - _already_in_tm

        st.info(f"This book has **{len(_corrected)} corrected sentences** "
                f"({_already_in_tm} already in TM, **{_new_count} new**).")

        if st.button(f"✅ Save All {_new_count} New Corrected Sentences to TM",
                     disabled=_new_count == 0, key="btn_batch_save_tm"):
            saved = 0
            for c in _corrected:
                if c["original_text"] not in _tm_data:
                    save_to_translation_memory(c["original_text"], c["translated_text"])
                    saved += 1
            reload_translation_memory()
            st.success(f"Saved {saved} new sentence pairs to translation_memory.json!")
            st.rerun()

        st.markdown("---")

        # ── PATH 2: CSV Import ───────────────────────────────────────────────
        st.markdown("### Path 2 — Bulk Import from CSV / Spreadsheet")
        st.write(
            "Prepare a spreadsheet with two columns: **Gujarati** (col A) and **Bengali** (col B). "
            "No header row needed. Export as CSV and upload here."
        )
        with st.expander("📋 Template format"):
            st.code(
                "ભ-aw.,ভ-aw.\n"
                "ગઢડા પ્રથ-aw.,গড়দা প্রথ-aw.\n"
                "...",
                language="text"
            )
            st.markdown("Or download the current TM as a starting point:")
            _tm_csv_lines = [f"{k},{v}" for k, v in _tm_data.items()]
            st.download_button(
                "⬇️ Download TM as CSV",
                data="\n".join(_tm_csv_lines).encode("utf-8"),
                file_name="translation_memory_export.csv",
                mime="text/csv"
            )

        _uploaded_csv = st.file_uploader("Upload CSV (Gujarati, Bengali)", type=["csv"], key="tm_csv_upload")
        if _uploaded_csv:
            import csv as _csv
            import io as _io
            content = _uploaded_csv.read().decode("utf-8")
            reader = _csv.reader(_io.StringIO(content))
            rows = [(r[0].strip(), r[1].strip()) for r in reader
                    if len(r) >= 2 and r[0].strip() and r[1].strip()]
            new_rows = [(g, b) for g, b in rows if g not in _tm_data]
            st.info(f"CSV has {len(rows)} pairs → {len(new_rows)} are new (not yet in TM).")
            if new_rows:
                st.dataframe({"Gujarati": [g[:60] for g, _ in new_rows[:10]],
                              "Bengali":  [b[:60] for _, b in new_rows[:10]]},
                             use_container_width=True)
                if len(new_rows) > 10:
                    st.caption(f"...and {len(new_rows)-10} more")
                if st.button(f"⬆️ Import {len(new_rows)} New Pairs into TM", key="btn_csv_import"):
                    for g, b in new_rows:
                        save_to_translation_memory(g, b)
                    reload_translation_memory()
                    st.success(f"Imported {len(new_rows)} pairs into translation_memory.json!")
                    st.rerun()

        st.markdown("---")

        # ── PATH 3: Cloud-Assisted Batch Generation ──────────────────────────
        st.markdown("### Path 3 — Cloud-Assisted Batch Generation")
        st.write(
            "Send untranslated or poorly-translated sentences to **Gemini** for high-quality "
            "draft translations, then approve or edit each one before saving to TM. "
            "Much faster than translating from scratch."
        )
        _cloud_key_tm = st.text_input("Gemini API Key", type="password", key="tm_gemini_key")
        _batch_n = st.slider("Sentences to generate per batch", 5, 50, 20, key="tm_batch_n")

        # Pick sentences not yet in TM (from current book's pending/done chunks)
        _not_in_tm = [c for c in _all_chunks
                      if c["original_text"].strip()
                      and c["original_text"] not in _tm_data
                      and len(c["original_text"]) > 15]

        st.info(f"{len(_not_in_tm)} sentences in this book are not yet in TM.")

        if "tm_cloud_drafts" not in st.session_state:
            st.session_state.tm_cloud_drafts = []

        if st.button(f"⚡ Generate {_batch_n} Draft Translations via Gemini",
                     disabled=not _cloud_key_tm or not _not_in_tm, key="btn_tm_cloud_gen"):
            import google.generativeai as _genai
            _genai.configure(api_key=_cloud_key_tm)
            _model_g = _genai.GenerativeModel("gemini-2.0-flash")
            _batch_texts = [c["original_text"] for c in _not_in_tm[:_batch_n]]
            _prompt = (
                "You are an expert translator of BAPS Swaminarayan religious texts. "
                "Translate each Gujarati sentence to Bengali, preserving the devotional "
                "tone and BAPS-specific vocabulary (Vachanamrut→বচনামৃত, "
                "Maharaj→মহারাজ, Sant→সন্ত, Gunatit→গুণাতীত). "
                "Use respectful/honorific Bengali forms for God and saints. "
                "Return ONLY a JSON array of translated strings in the same order.\n\n"
                f"Input: {json.dumps(_batch_texts, ensure_ascii=False)}"
            )
            with st.spinner(f"Generating {_batch_n} translations via Gemini..."):
                try:
                    _resp = _model_g.generate_content(
                        _prompt, generation_config={"response_mime_type": "application/json"}
                    )
                    _drafts = json.loads(_resp.text)
                    if isinstance(_drafts, list) and len(_drafts) == len(_batch_texts):
                        st.session_state.tm_cloud_drafts = list(zip(_batch_texts, _drafts))
                        st.success(f"Generated {len(_drafts)} drafts — review below!")
                    else:
                        st.error("Unexpected response format from Gemini.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if st.session_state.tm_cloud_drafts:
            st.markdown("#### Review Generated Drafts")
            st.write("Edit any Bengali translation, then click **Save Selected** to add to TM.")
            _approved = []
            for idx, (guj, ben) in enumerate(st.session_state.tm_cloud_drafts):
                col_g, col_b, col_ok = st.columns([4, 4, 1])
                with col_g:
                    st.text_area("Gujarati", guj, height=80, disabled=True,
                                 key=f"tm_guj_{idx}")
                with col_b:
                    edited_ben = st.text_area("Bengali (edit if needed)", ben,
                                              height=80, key=f"tm_ben_{idx}")
                with col_ok:
                    st.markdown("<br>", unsafe_allow_html=True)
                    approved = st.checkbox("✓", value=True, key=f"tm_ok_{idx}")
                if approved:
                    _approved.append((guj, edited_ben))

            if st.button(f"💾 Save {len(_approved)} Approved Pairs to TM", key="btn_tm_save_drafts"):
                for g, b in _approved:
                    save_to_translation_memory(g, b)
                reload_translation_memory()
                st.session_state.tm_cloud_drafts = []
                st.success(f"Saved {len(_approved)} pairs! TM now has {len(_tm_data) + len(_approved)} entries.")
                st.rerun()

        st.markdown("---")

        # ── PATH 4: File Alignment (Gujarati + Bengali Book) ─────────────────
        st.markdown("### Path 4 — Align Existing Gujarati + Bengali Book Files ⭐ FASTEST")
        st.write(
            "If you have a **Gujarati Vachanamrut** AND its **Bengali translation** "
            "(official BAPS publication), upload both. The tool aligns them sentence by "
            "sentence and bulk-adds all pairs to TM. This can add **1,000+ entries at once**."
        )
        st.info("💡 Supports PDF, EPUB, and TXT for both files.")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            _guj_file = st.file_uploader("Gujarati Source Book",
                                         type=["txt", "pdf", "epub"], key="align_guj")
        with col_a2:
            _ben_file = st.file_uploader("Bengali Translation Book",
                                         type=["txt", "pdf", "epub"], key="align_ben")

        def _extract_sentences_from_upload(uploaded, lang_hint="guj"):
            """Extract and split an uploaded file into sentences."""
            import tempfile
            ext = os.path.splitext(uploaded.name)[1].lower()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            if ext in (".pdf", ".epub"):
                chapters, _, _ = extract_text_from_file(tmp_path)
                full_text = " ".join(text for _, text in chapters)
            else:
                full_text = uploaded.getvalue().decode("utf-8", errors="replace")
            os.unlink(tmp_path)
            # Split into sentences
            sents = [s.strip()
                     for s in re.split(r'(?<=[.!?।॥])\s+', full_text)
                     if s.strip() and len(s.strip()) > 15]
            return sents

        if _guj_file and _ben_file:
            with st.spinner("Extracting and aligning sentences..."):
                _guj_sents = _extract_sentences_from_upload(_guj_file, "guj")
                _ben_sents = _extract_sentences_from_upload(_ben_file, "ben")

            _pairs = min(len(_guj_sents), len(_ben_sents))
            st.success(
                f"Extracted **{len(_guj_sents)} Gujarati** and **{len(_ben_sents)} Bengali** sentences. "
                f"Will align **{_pairs} pairs** by position."
            )

            if _pairs > 0:
                _already = sum(1 for g in _guj_sents[:_pairs] if g in _tm_data)
                _new_align = _pairs - _already
                st.info(f"**{_new_align} pairs are new** (not yet in TM).")

                st.markdown("**Preview of first 15 aligned pairs:**")
                st.dataframe({
                    "Gujarati": [g[:80] for g in _guj_sents[:15]],
                    "Bengali":  [b[:80] for b in _ben_sents[:15]],
                }, use_container_width=True)

                if st.button(f"📥 Import All {_new_align} New Pairs to TM",
                             disabled=_new_align == 0, key="btn_align_import"):
                    with st.spinner(f"Importing {_new_align} pairs..."):
                        saved_align = 0
                        for g, b in zip(_guj_sents[:_pairs], _ben_sents[:_pairs]):
                            if g not in _tm_data:
                                save_to_translation_memory(g, b)
                                saved_align += 1
                    reload_translation_memory()
                    st.success(f"✅ Imported {saved_align} pairs! TM now has {len(_tm_data) + saved_align} entries.")
                    st.rerun()

        st.markdown("---")
        # ── Current TM contents ───────────────────────────────────────────────
        with st.expander(f"📋 View all {len(_tm_data)} TM entries"):
            _tm_df = {"Gujarati": list(_tm_data.keys()),
                      "Bengali":  list(_tm_data.values())}
            st.dataframe(_tm_df, use_container_width=True, height=400)

    # ----------------- TAB 4: Glossary Management -----------------
    with tab_glossary:
        st.subheader("🏷️ Custom Terminology Glossary")
        st.write("Register custom dictionary mappings to protect key terms (e.g. BAPS guru names, specific places).")
        
        # Form to add a glossary rule
        with st.form("glossary_tab_form", clear_on_submit=True):
            col_rule1, col_rule2 = st.columns(2)
            with col_rule1:
                g_orig = st.text_input("Source Word", placeholder="e.g. સ્વામિનારાયણ")
            with col_rule2:
                g_trans = st.text_input("Bengali Target Word", placeholder="e.g. স্বামীনারায়ণ")
            
            submit_g = st.form_submit_button("➕ Register Terminology Rule")
            if submit_g and g_orig and g_trans:
                db_helper.save_glossary_term(g_orig, g_trans)
                st.session_state.glossary_cache = load_glossary(db_helper)
                st.success(f"Added glossary entry: '{g_orig}' ➔ '{g_trans}'")
                st.rerun()
                
        # Display existing rules
        glossary_data = db_helper.get_glossary()
        st.session_state.glossary_cache = glossary_data
        
        st.markdown("### 📋 Active Terminology Rules")
        if glossary_data:
            g_df = pd.DataFrame(list(glossary_data.items()), columns=["Source Word", "Bengali Word"])
            
            # Loop to render clean deleteable row layout
            for idx, row in g_df.iterrows():
                col_w1, col_w2, col_w3 = st.columns([4, 4, 1])
                with col_w1:
                    st.text(row["Source Word"])
                with col_w2:
                    st.text(row["Bengali Word"])
                with col_w3:
                    if st.button("🗑️", key=f"del_tab_{row['Source Word']}"):
                        db_helper.delete_glossary_term(row["Source Word"])
                        st.session_state.glossary_cache = load_glossary(db_helper)
                        st.toast("Glossary rule deleted!", icon="🗑️")
                        st.rerun()
        else:
            st.info("No custom terminology rules configured. Add rules above to bypass translation mapping for specific words.")
