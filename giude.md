Gujarati → Bengali
Book Translator
A Complete Build Guide for AI/Software Engineering Students
Fully local · Zero cost · 100% private
📋
 Quick overview  
Who this is for: Bachelor's student in AI or Software Engineering
Prerequisites: Basic Python knowledge, able to run commands in a terminal
Time to build: 1–2 weekends
Hardware needed: Any laptop/desktop with 8 GB RAM (no GPU required)
Total cost: Free
1. What you are building
You will build a desktop application that translates entire books — PDFs, EPUBs, or plain text files — from Gujarati to 
Bengali. Everything runs on your own computer. No data is ever sent to any server. No API keys, no subscriptions, no 
cost.
How it works (big picture)
● User drops a book file into a simple web interface
● The app extracts all text and splits it into sentence-sized chunks
● Each chunk is sent to IndicTrans2, a free Indian-language AI model running locally
● Translated chunks are reassembled in the correct order
● User downloads the translated Bengali file
💡 Why IndicTrans2 and not Ollama/ChatGPT?
IndicTrans2 was built specifically for all 22 scheduled Indian languages by AI4Bharat (IIT Madras). It translates 
Gujarati → Bengali directly, without converting to English first. This gives much better quality for literary/book 
text than general-purpose models like Llama or Gemma.
2. Tools and technologies
Core stack
● Python 3.10+ — main programming language
● IndicTrans2 (AI4Bharat) — translation model, MIT licensed, free
● IndicTransToolkit — official helper library for IndicTrans2
● HuggingFace Transformers — loads and runs the model
● PyMuPDF (fitz) — extracts text from PDF files
● ebooklib — extracts text from EPUB files
● Streamlit — builds the web UI (runs in your browser, still local)
● SQLite — saves progress so you can resume large books
AI coding assistant
Use Claude (claude.ai) or GitHub Copilot to write the code. This guide gives you the full architecture and prompts — the 
AI assistant fills in the implementation details. You will understand every file you create.
3. Project structure
Create a folder called book-translator. Inside it, create this structure:
book-translator/
├── app.py                  ← Streamlit UI
├── translator.py           ← core translation logic
├── extractor.py            ← PDF/EPUB text extraction
├── chunker.py              ← sentence splitting
├── progress.py             ← SQLite resume tracking
├── assembler.py            ← rebuild output file
├── requirements.txt        ← Python dependencies
└── models/                 ← model weights stored here (auto-downloaded)
Each file has a single clear responsibility. You will build them one at a time.
4. Step-by-step build instructions
Step 1    Set up your Python environment
Open your terminal and run these commands:
# Install Python 3.10 or newer if you don't have it
python --version
# Create a virtual environment (keeps your project isolated)
cd book-translator
python -m venv venv
# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
# Your terminal should now show (venv) at the start
Step 2    Install dependencies
Create a file called requirements.txt with this content:
torch
transformers
sentencepiece
sacremoses
indic-transliteration
IndicTransToolkit
PyMuPDF
ebooklib
beautifulsoup4
streamlit
tqdm
Then install everything:
pip install -r requirements.txt
  ⏱This will take 5–10 minutes. PyTorch is large (~800 MB). Only needs to be done once.
If you get errors on Windows, try: pip install torch --index-url https://download.pytorch.org/whl/cpu
Step 3    Build the text extractor (extractor.py)
This file handles reading text from PDF, EPUB, and plain text files.
Paste this into Claude and ask it to write the code (see Section 6 for the prompt):
extractor.py should:
• Accept a file path and detect the file type automatically
• For PDF: use PyMuPDF to extract text page by page, preserving paragraph breaks
• For EPUB: use ebooklib + BeautifulSoup to extract chapter text in order
• For TXT: read directly
• Return a list of (chapter_title, chapter_text) tuples
• Handle encoding errors gracefully
Step 4    Build the sentence chunker (chunker.py)
IndicTrans2 works best on sentences or short paragraphs — not whole pages at once.
chunker.py should:
• Take a string of text as input
• Split on sentence boundaries (periods, question marks, exclamation marks — also Gujarati punctuation: । )
• Never split a chunk longer than 400 characters; break at word boundaries
• Return a list of clean string chunks
• Preserve blank lines that indicate paragraph breaks (store them as empty string markers)
Step 5    Build the progress tracker (progress.py)
Books can be 300+ pages. If your computer sleeps or the app crashes, you need to resume from where you left off — not 
start again.
progress.py should:
• Create a SQLite database file named after the book (e.g. mybook_progress.db)
• Store each chunk with: chunk_id, original_text, translated_text, status (pending/done)
• Provide: save_chunk(id, original, translated), get_pending_chunks(), get_all_chunks()
• On startup, if a database for this book already exists, skip completed chunks automatically
Step 6    Build the translator (translator.py)
This is the most important file. It loads IndicTrans2 and runs translations.
translator.py should:
• Load the model ai4bharat/indictrans2-indic-indic-dist-200M from HuggingFace (distilled = faster on CPU)
• Set source language to guj_Gujr and target language to ben_Beng
• Accept a list of text chunks as input
• Translate them in small batches of 8 at a time (reduces memory use)
• Show a tqdm progress bar
• Return translated chunks in the same order
• Include a translate_book() function that ties together: load progress DB → get pending chunks → translate → 
save each result immediately
⚠️ CPU performance note: The distilled 200M model takes roughly 20–40 seconds per page on a modern CPU 
laptop. A 200-page book will take 1–2 hours. Let it run in the background.
The model weights (~800 MB) are downloaded automatically the first time. After that, everything is offline.
Step 7  
  Build the output assembler (assembler.py)
Puts translated chunks back together into a readable output file.
assembler.py should:
• Accept a list of (chapter_title, [translated_chunks]) tuples
• Write a clean Bengali text file with chapter headings and proper paragraph spacing
• Also offer a basic PDF output option using reportlab (optional, add last)
• Handle Bengali Unicode correctly — use encoding='utf-8' everywhere
• Return the output file path
Step 8  
  Build the Streamlit UI (app.py)
Streamlit turns your Python script into a web app that runs in your browser. No HTML or CSS needed.
app.py should:
• Show a file upload widget that accepts PDF, EPUB, TXT
• Show a 'Start Translation' button
• Display a live progress bar (st.progress) updated per chunk
• Show estimated time remaining
• When done, show a Download button for the Bengali output file
• Show a side-by-side preview of 3 original vs translated sentences
• If a progress database exists for the uploaded file, ask: Resume or Start Fresh?
5. Running the app (daily use instructions)
First time setup (once only)
cd book-translator
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt  # only first time
streamlit run app.py
Your browser will open automatically at http://localhost:8501
Every day after that
cd book-translator
source venv/bin/activate
streamlit run app.py
That's it — three commands.
Translating a book — step by step
● Open http://localhost:8501 in your browser
● Click 'Browse files' and upload your Gujarati PDF, EPUB, or TXT
● The app detects the file and shows how many chapters/pages it found
● Click 'Start Translation'
● Watch the progress bar — each chunk updates in real time
● Go make tea. A 200-page book takes 1–2 hours on CPU
● When done, click 'Download Bengali Translation'
● The output is saved in the book-translator/output/ folder too
Resuming an interrupted translation
● Just upload the same file again
● The app detects the existing progress database
● Click 'Resume' — it picks up exactly where it stopped
💾
 Important  
Never delete the .db file in the book-translator folder while a translation is in progress.
You can safely close the browser tab — the translation continues in the terminal.
To stop translation: press Ctrl+C in the terminal. Resume any time.
6. How to use Claude AI to write the code
You do not need to write all the code yourself. Use Claude (claude.ai — free tier works) as your pair programmer. Here 
is exactly how:
The master prompt — use this first
Copy and paste this into a new Claude conversation before you start:
I am a bachelor's student in AI/Software Engineering building a Gujarati to Bengali book translator. It runs 
100% locally using IndicTrans2 (ai4bharat/indictrans2-indic-indic-dist-200M from HuggingFace). No GPU — 
CPU only. The stack is: Python 3.10, HuggingFace Transformers, PyMuPDF, ebooklib, Streamlit, SQLite for 
resume tracking.
Project files: app.py (Streamlit UI), translator.py (IndicTrans2 wrapper), extractor.py (PDF/EPUB/TXT 
reading), chunker.py (sentence splitting), progress.py (SQLite tracking), assembler.py (output builder).
Rules: use IndicTransToolkit for preprocessing, batch size 8 for CPU, source lang guj_Gujr, target lang 
ben_Beng. Always use utf-8 encoding. Add docstrings to every function. Keep each file under 150 lines.
I will ask you to build one file at a time. Always give me complete, runnable code.
Per-file prompts — use these one at a time
After setting up context with the master prompt, send one of these for each file:
For extractor.py:
Write extractor.py. It should auto-detect PDF, EPUB, or TXT by file extension. For PDF use PyMuPDF (fitz) to 
extract text page by page with paragraph breaks. For EPUB use ebooklib + BeautifulSoup4 to get chapters in 
reading order. Return a list of (chapter_title: str, chapter_text: str) tuples. Handle encoding errors with 
errors='replace'.
For chunker.py:
Write chunker.py. Input: a string. Output: list of strings. Split on sentence endings including Gujarati danda (।). 
Max chunk size 400 characters; break at word boundary if needed. Preserve paragraph breaks as empty string 
entries in the list.
For progress.py:
Write progress.py using SQLite3 (built into Python, no install needed). Class ProgressDB. Methods: 
__init__(book_path) creates DB named after the book, save_chunk(chunk_id, original, translated, status), 
get_pending_chunks() returns list of (id, text) not yet translated, mark_done(chunk_id, translated_text), 
get_all_translated() returns ordered list of translated strings, is_complete() returns bool.
For translator.py:
Write translator.py using HuggingFace Transformers and IndicTransToolkit. Load model ai4bharat/indictrans2
indic-indic-dist-200M. Source: guj_Gujr, target: ben_Beng. Function translate_chunks(chunks: list[str]) -> 
list[str] processes in batches of 8 with tqdm progress bar. Function translate_book(book_path, chapters) 
coordinates with ProgressDB — skip already translated chunks, save each result immediately after translation.
For assembler.py:
Write assembler.py. Function assemble_output(chapters: list[tuple[str, list[str]]], output_path: str). Writes a 
UTF-8 text file with chapter headings in bold-equivalent (use === heading === markers), Bengali paragraph 
text with double newlines between paragraphs. Return the output file path.
For app.py:
Write app.py using Streamlit. Features: file uploader (PDF/EPUB/TXT), detect existing progress DB and offer 
Resume/Fresh choice, Start Translation button, real-time progress bar with chunk count and % complete, 
estimated minutes remaining (based on average chunk time), side-by-side table showing 3 sample original vs 
translated sentences, Download button when complete. Import and use extractor, chunker, translator, assembler 
modules.
When something doesn't work
Copy the error message and send this to Claude:
I got this error when running [filename]: [paste error here]. I am using Python 3.10, CPU only, IndicTrans2 
distilled model. Here is my current code: [paste your code]. What is wrong and how do I fix it?
7. Testing your build
Before translating a full book, test with small samples.
Quick sanity test
Create a file called test.py and run it after building translator.py:
from translator import translate_chunks
test_sentences = [
    "     ગુજરાત ભારાતનુંએક સુંદરા રાજ્ય છે.",
    "     આ પુસ્તક ખૂબ જ રાસુંપુદ છે.",
]
results = translate_chunks(test_sentences)
for orig, trans in zip(test_sentences, results):
    print(f"GUJ: {orig}")
    print(f"BEN: {trans}")
    print()
Expected output: Bengali text that says Gujarat is a beautiful state of India / This book is very interesting.
Test the full pipeline
● Create a small 2-page PDF in Gujarati (or find one online)
● Upload it through the Streamlit UI
● Verify the progress bar moves and chunks are being saved
● Check the output file opens correctly with Bengali Unicode text
8. Troubleshooting common issues
Problem
Fix
Model download fails
Check internet connection. Try: export 
HF_HUB_OFFLINE=0. If behind a proxy, set 
HTTPS_PROXY environment variable.
'sentencepiece not found'
Run: pip install sentencepiece protobuf
Memory error (RAM)
Reduce batch size from 8 to 4 in translator.py. Or close 
other apps.
Bengali text shows as boxes/?
Your output file is correct (UTF-8). The problem is your 
PDF viewer. Open the .txt output in Notepad++ or VS 
Code instead.
Streamlit shows blank page
Clear browser cache. Or open http://localhost:8501 in a 
private window.
Translation is very slow
Normal on CPU. The distilled 200M model is the fastest 
option. Let it run overnight for large books.
EPUB chapter order wrong
In extractor.py, sort chapters by spine order from ebooklib: 
book.spine gives the correct sequence.
9. Speeding up translation with cloud LLMs
The local IndicTrans2 model is free and fully private, but slow on CPU (1–2 hours for a 200-page book). If you need 
faster results, or better literary quality, you can optionally route some or all of the work through a cloud LLM. This 
section explains the trade-offs and how to build it.
  ⚠️Privacy trade-off — read this first
Cloud LLMs (Claude API, OpenAI, Gemini) send text to a company server. For a published book this is usually 
fine. For private or confidential documents, stay 100% local with IndicTrans2. Design your app so the user 
chooses: 'Local (private, slow)' vs 'Cloud (fast, leaves your machine)'.
Understanding exactly what is and isn't shared (Claude API)
Before adding any cloud option, understand precisely what happens to your text. This matters — don't guess, and don't 
let anyone using your tool guess either. Here is the honest, complete picture based on Anthropic's current published terms 
(verify at privacy.claude.com before relying on it, as policies change).
The single most important fact:
  🔑With local IndicTrans2, your text NEVER leaves your computer — nobody can see it because it is never 
sent anywhere.
With the Claude API, your text IS sent to Anthropic's servers to be processed. This is unavoidable for any cloud 
model — the computation happens on their machines, not yours. 'Cloud' always means the text travels off your 
machine. That is the core difference between the two options.
What Anthropic does and does not do with API text (default commercial terms):
Question Answer (default API terms)
Is my text used to train future models? No — not by default. Anthropic's Commercial Terms say API 
content is not used for training unless you explicitly opt into a 
program. Book translation does not opt you in.
How long is my text stored? Up to 30 days on Anthropic's backend, then automatically deleted. 
Not kept forever.
Can Anthropic employees read it? No, by default. Access happens only through a controlled, logged 
path (e.g. after an automated safety flag), and the access log is 
tamper-proof.
Can other customers or the public see it? No. It is never shared with other users, never sold, and never made 
public.
Is there any exception to the 30-day deletion? Yes — if a request is flagged by automated safety systems for a 
policy violation, inputs/outputs may be kept up to 2 years and 
safety scores up to 7 years. For ordinary book text this essentially 
never triggers, but it is a real carve-out you should know about.
Could a third party ever get it 'from anywhere'? Only through the same rare paths that apply to ANY cloud 
service: a lawful legal/government demand, or a security breach. 
Low probability, but never mathematically zero. The only way to 
make it truly zero is to not send the data — i.e. stay local.
So — is it safe to use?
  ✅For a book you intend to publish anyway: yes, the Claude API is a reasonable, low-risk choice. Your text 
isn't trained on, isn't kept long-term, isn't readable by staff in normal use, and isn't shared with anyone.
For private, confidential, or unpublished material that must never touch a third-party server: use local 
IndicTrans2 only. It is the single option where the text truly never leaves your machine.
Rule of thumb: 'If the book must never leave your computer, keep it local. If you'll publish it anyway, the API is 
fine.'
Note on Zero Data Retention (ZDR):
Anthropic offers a stricter ZDR arrangement where inputs/outputs are not stored at all (except what's needed to comply 
with law or combat misuse — safety classifier results are still kept). However, ZDR requires an approved commercial 
agreement and is not something a student gets automatically on a personal API key. For this project, assume the standard 
30-day default applies unless the student's organization has arranged ZDR.
Design implication for the app:
● Add a clear engine selector in the UI: 'Local — private, slower' vs 'Cloud — faster, sends text to Anthropic'
● Default to Local. Make the user actively choose Cloud, so the choice to send data off-machine is always deliberate
● Show a one-line reminder next to the Cloud option: 'Your text will be sent to Anthropic's servers to translate.'
Option A — Claude API (best quality, low cost)
The Claude API is excellent at Indian languages and preserves literary tone very well. It is a paid API but extremely 
cheap for text: translating a full book typically costs well under a dollar.
● Sign up at console.anthropic.com and create an API key
● Install the SDK: pip install anthropic
● Use the Batch API for book translation — it is about 50% cheaper and designed for large jobs that don't need 
instant results
● Recommended model: claude-sonnet-4-6 (great quality/cost balance for translation)
Add a new file claude_translator.py alongside your local translator.py:
claude_translator.py should:
• Read the API key from an environment variable ANTHROPIC_API_KEY (never hardcode keys)
• Function translate_chunks_claude(chunks: list[str]) -> list[str]
• Send each chunk with a clear system prompt: 'You are a professional literary translator. Translate the 
following Gujarati text to Bengali. Output only the Bengali translation, nothing else. Preserve tone, meaning, 
and paragraph structure.'
• Use max_tokens=1024, model claude-sonnet-4-6
• Save each result to the same ProgressDB so resume still works
• For large books, group 5–10 chunks per API call to reduce overhead and cost
Verify the current model names and Batch API details before you build — see console.anthropic.com/docs. Model names 
change over time.
Option B — Hybrid (recommended sweet spot)
The smartest design: use free local IndicTrans2 for the bulk of the book, and only send difficult passages to Claude API. 
This keeps cost near zero while fixing the weakest translations.
● Translate the whole book locally with IndicTrans2 first
● Add a confidence check: flag chunks that are very long, contain many untranslated characters, or where back
translation differs a lot from the original
● Send only those flagged chunks to Claude API for a better translation
● A typical book might send only 5–10% of chunks to the cloud — cost stays a few cents
💡 This hybrid pattern is genuinely good engineering and worth highlighting on your portfolio: 'Cost-optimised 
translation pipeline — 90% local/free, 10% cloud for quality-critical passages.'
Option C — Other LLM choices
Option
Cost
Private?
Best for
IndicTrans2 (local)
Free
Yes — 100%
Claude API
~$0.5–1/book
No
Gemini Flash API
Very low
No
Ollama + Navarasa (local)
Free
Yes — 100%
Google Gemini free tier
Free (limited)
Other ways to speed up (no cloud needed)
No
Default. Private books, zero 
budget, offline use
Highest literary quality, 
published books
Cheapest cloud option, high 
volume/bulk
Free literary refinement pass, 
still offline
Testing / small books within 
free quota
● Parallel processing: use Python multiprocessing to run several CPU translation workers at once — can cut time 
significantly on multi-core machines
● Use the 200M distilled model (already recommended) rather than the 1B model — roughly 3–4x faster on CPU
● Deduplicate repeated sentences before translating — books often repeat phrases; translate each unique chunk once 
and reuse the result
● Run overnight: for a fully private, free workflow, just start the local translation before bed
Prompt to give Claude for building the cloud option
Add a claude_translator.py module to my Gujarati→Bengali book translator. It should mirror my existing 
translator.py interface (function translate_chunks(chunks: list[str]) -> list[str]) so I can swap engines easily. Use 
the anthropic Python SDK, read the key from ANTHROPIC_API_KEY env var, model claude-sonnet-4-6, and a 
system prompt that instructs professional literary Gujarati-to-Bengali translation outputting only Bengali. Group 
8 chunks per request for efficiency. Save results to my existing ProgressDB. Show me the code and tell me how 
to set the API key as an environment variable on Windows and Mac.
10. Enhancements (once the basic version works)
After your core build is working, here are improvements to add — each is a good mini-project:
● PDF output with Bengali font: use reportlab with a Bengali TTF font (Noto Sans Bengali, free from Google Fonts) 
to produce a properly formatted Bengali PDF instead of plain text
● Quality scoring: after each translation, run a simple BLEU check by back-translating a sample chunk Bengali → 
English using IndicTrans2's en-indic model and checking if the meaning is preserved
● Glossary support: let users define a CSV of specific words/names (e.g. character names) that should not be 
translated — inject these as pre/post-processing rules
● Batch folder mode: translate all books in a folder overnight with one command
● Navarasa 2.0 refinement pass: after IndicTrans2 translation, optionally run chunks through Navarasa 2.0 via Ollama 
for literary tone improvement
11. Reference: key links
● IndicTrans2 GitHub: https://github.com/AI4Bharat/IndicTrans2
● IndicTransToolkit (PyPI): pip install IndicTransToolkit
● HuggingFace model page: huggingface.co/ai4bharat/indictrans2-indic-indic-dist-200M
● Streamlit docs: docs.streamlit.io
● PyMuPDF docs: pymupdf.readthedocs.io
● Noto Sans Bengali font (free): fonts.google.com/noto/specimen/Noto+Sans+Bengali
● Claude AI (for coding help): claude.ai
● Claude API console (optional cloud speedup): console.anthropic.com
🎓
 Final note  
Good luck! This is a genuinely useful project — book translation for Indian languages is a real 
gap. If you build this well, it is worth putting on your portfolio and GitHub.
Suggested GitHub repo name: indic-book-translator
Suggested README tagline: 'Free, private, offline Gujarati → Bengali book translator using IndicTrans2'