# 📚 BAPS Book Translator

A Premium Multi-Source to Bengali Literary Translation Suite supporting Local & Cloud Translation Engines.

BAPS Book Translator is a streamlined, feature-rich web application built with Streamlit. It allows users to translate large books and documents (PDF, EPUB, TXT) from languages like Gujarati, Hindi, and English into Bengali.

## 🌟 Key Features

- **Multi-Format Support**: Upload `.pdf`, `.epub`, or `.txt` files directly.
- **Multiple Translation Engines**: 
  - **Local (IndicTrans2)**: 100% offline and private translation running on your local machine (ideal for privacy-sensitive documents).
  - **Cloud (Gemini / Claude)**: Fast and highly accurate translation utilizing powerful online AI models.
- **Side-by-Side Editor**: Interactive workspace to review the translation line-by-line and make manual corrections on the fly.
- **Glossary Management**: Register custom terminology rules (e.g., specific name or term mappings) to bypass automated translation and maintain consistency.
- **Progress Tracking**: Pausable and resumable translation batches. Automatically tracks translated chunks, pending chunks, and user corrections.
- **Multi-Format Export**: Once translated, reassemble and download the output book in `.txt`, `.epub`, or `.pdf` (with embedded Bengali fonts) formats.

## 🛠️ Installation & Setup

1. **Navigate to the project directory**:
   ```bash
   cd BAPS_Book_Translator
   ```

2. **Install the dependencies**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This project uses PyTorch, Transformers, Streamlit, and several other libraries for document parsing and UI.)*

3. **Run the Application**:
   Launch the web app using Streamlit:
   ```bash
   streamlit run app.py
   ```

## 🚀 How to Use

1. **Configure Settings**: Open the left sidebar to select your **Source Language** (Gujarati, Hindi, English) and your **Translation Engine**. 
   - If using cloud engines, enter your Gemini or Claude API key.
   - If using the local IndicTrans2 engine, it will cache the model locally.
2. **Upload a Book**: Drag and drop your document. The application will chunk the text and initialize a local SQLite database to track your progress.
3. **Translate**: Click **Start Translation**. You can pause and resume at any time. A live status tracker will estimate the time remaining.
4. **Edit & Review**: Switch to the **Side-by-Side Editor** tab to manually correct any sentences. Add specific rules in the **Terminology Glossary** tab.
5. **Export**: Once 100% complete, head to the "Reassemble & Download Book" section to export your translated masterpiece as a TXT, EPUB, or PDF file.

## 📁 Project Structure

- `app.py`: Main Streamlit web application dashboard and UI.
- `extractor.py`: Handles text extraction from PDF, EPUB, and TXT files.
- `chunker.py`: Splits large texts into manageable chunks for accurate translation.
- `translator.py`: Contains API integrations for Gemini, Claude, and local IndicTrans2 models.
- `progress.py`: Manages the local SQLite database to save translation states, user edits, and glossary terms.
- `assembler.py`: Re-compiles translated chunks back into the chosen export formats.
- `requirements.txt`: Python package dependencies.
