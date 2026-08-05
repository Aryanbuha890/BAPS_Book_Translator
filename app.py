import os
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
    check_translation_memory
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
        help="Local runs 100% offline & privately on your CPU. Cloud requires internet but translates faster."
    )
    
    # Check compatibility/download state for local engine
    if engine == "Local (IndicTrans2)":
        if src_lang == "eng_Latn":
            model_dir = os.path.join("models", "models--ai4bharat--indictrans2-en-indic-1B")
            if not os.path.isdir(model_dir):
                st.warning("⚠️ Local English-to-Bengali translation requires the English-to-Indic model. Please run `python download_model.py` and select Option 2 or 3 to download the weights.")
        else:
            model_dir = os.path.join("models", "models--ai4bharat--indictrans2-indic-indic-1B")
            if not os.path.isdir(model_dir):
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
    "Upload Document (PDF, EPUB, TXT)", 
    type=["pdf", "epub", "txt"],
    help="Upload the book file you want to translate."
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
        
        # Extract and parse file
        with st.spinner("Extracting content and parsing chapters..."):
            try:
                 chapters, detected_lang = extract_text_from_file(temp_path)
                
                 total_text_length = sum(len(text.strip()) for _, text in chapters) if chapters else 0
                 if total_text_length == 0:
                     st.error("⚠️ No text could be extracted from this document. It might be a scanned PDF or empty. Please upload a text-based PDF or convert it using OCR first.")
                     st.session_state.book_path = None
                     st.session_state.current_db = None
                     st.session_state.detected_lang = None
                     st.session_state.detected_lang_label = None
                     st.session_state.file_hash = None
                 else:
                     # Cache auto-detected language
                     lang_labels = {"guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "eng_Latn": "English", "unknown": "Unknown"}
                     st.session_state.detected_lang = detected_lang
                     st.session_state.detected_lang_label = lang_labels.get(detected_lang, "Unknown")
                     
                     chapters_chunked = []
                     for title, text in chapters:
                         chunks = chunk_text(text)
                         chapters_chunked.append((title, chunks))
                         
                     # Compare new chunks with DB to see if the file changed
                     stats = db_helper.get_progress_stats()
                     if stats['total_chunks'] > 0:
                         db_chunks = [c['original_text'] for c in db_helper.get_all_chunks()]
                         new_chunks = [chunk for _, chunks in chapters_chunked for chunk in chunks]
                         if db_chunks != new_chunks:
                             with db_helper.conn:
                                 db_helper.conn.execute("DELETE FROM translation_progress")
                                 
                     # Initialize DB (resets DB if chunk count is mismatched with new file)
                     is_new = db_helper.initialize_chunks(chapters_chunked)
                     if is_new:
                         st.toast("Success! Initialized new book translation database.", icon="✨")
                     else:
                         st.toast("Welcome back! Loaded existing translation progress.", icon="♻️")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.book_path = None
                st.session_state.current_db = None

# Main content dashboard when book is loaded
if st.session_state.current_db:
    db_helper = st.session_state.current_db
    stats = db_helper.get_progress_stats()
    st.session_state.stats = stats
    
    # Initialize workspace tabs
    tab_dash, tab_editor, tab_glossary = st.tabs([
        "📖 Translation Dashboard", 
        "✍️ Side-by-Side Editor", 
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
                            model_dir = os.path.join("models", "models--ai4bharat--indictrans2-en-indic-1B")
                            if not os.path.isdir(model_dir):
                                st.error("Local English-to-Bengali model weights not found. Please run `python download_model.py` and select Option 2 or 3 first.")
                            else:
                                st.session_state.translating = True
                                st.rerun()
                        else:
                            model_dir = os.path.join("models", "models--ai4bharat--indictrans2-indic-indic-1B")
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
                # Completely empty DB table to allow re-initialization of new chunks
                with db_helper.conn:
                    db_helper.conn.execute("DELETE FROM translation_progress")
                st.session_state.translating = False
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
                        translated_batch = translate_chunks_gemini(swapped_texts, api_key=api_key, src_lang=src_lang, tgt_lang="ben_Beng", glossary=st.session_state.glossary_cache)
                    else:
                        translated_batch = translate_chunks_claude(swapped_texts, api_key=api_key, src_lang=src_lang, tgt_lang="ben_Beng", glossary=st.session_state.glossary_cache)
                        
                    # Apply Glossary Swap-out
                    final_translated = []
                    for trans_t, p_map in zip(translated_batch, placeholder_maps):
                        r_text = swap_out(trans_t, p_map)
                        final_translated.append(r_text)
                        
                    # Save translations
                    for c_id, trans_txt in zip(batch_ids, final_translated):
                        db_helper.save_chunk(c_id, trans_txt)
                        
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
        st.write("Double-click any cell in the **Translated Bengali (Editable)** column to make adjustments. Changes are saved instantly.")
        
        # Load all chunks for data editor
        raw_chunks = db_helper.get_all_chunks()
        if raw_chunks:
            df_editor = pd.DataFrame(raw_chunks)
            
            # Create a copy to track updates
            edited_df = st.data_editor(
                df_editor,
                column_config={
                    "chunk_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "chapter_index": None, # Hide
                    "chapter_title": st.column_config.TextColumn("Chapter", disabled=True, width="medium"),
                    "original_text": st.column_config.TextColumn("Original Text", disabled=True, width="large"),
                    "translated_text": st.column_config.TextColumn("Translated Bengali (Editable)", width="large"),
                    "status": st.column_config.TextColumn("Status", disabled=True),
                    "modified_by_user": None # Hide
                },
                width="stretch",
                num_rows="fixed",
                key="sentence_editor"
            )
            
            # Save edited values back to database if edited
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

    # ----------------- TAB 3: Glossary Management -----------------
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
