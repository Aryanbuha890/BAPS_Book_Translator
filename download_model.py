import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODELS = {
    "a": ("facebook/nllb-200-distilled-600M",  "nllb-200-distilled-600M",  "~2.5 GB", "4 GB RAM",  "Low-end  (4–8 GB RAM, CPU-only OK)"),
    "b": ("facebook/nllb-200-distilled-1.3B",  "nllb-200-distilled-1.3B",  "~5 GB",   "8 GB RAM",  "Mid-range (8–16 GB RAM)"),
    "c": ("facebook/nllb-200-3.3B",             "nllb-200-3.3B",             "~13 GB",  "14 GB RAM", "High-end  (16+ GB RAM, GPU/MPS recommended)"),
}

def download_nllb_variant(repo_id: str, local_name: str) -> bool:
    from huggingface_hub import snapshot_download
    local_dir = os.path.join("models", local_name)
    print(f"\n--- Downloading '{repo_id}' → {local_dir} ---")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"✓ {repo_id} downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def download_indictrans(model_name: str, token: str) -> bool:
    print(f"\n--- Downloading: '{model_name}' ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="models", token=token)
        print(f"✓ Tokenizer downloaded.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="models", token=token)
        print(f"✓ Model weights downloaded.")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading '{model_name}': {e}")
        print(f"  Check: https://huggingface.co/{model_name} (must accept terms)")
        return False

def detect_recommended():
    """Suggests NLLB tier based on available RAM."""
    try:
        import subprocess
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        ram_gb = int(result.stdout.split(':')[1].strip()) / 1e9
    except Exception:
        ram_gb = 8  # conservative fallback
    if ram_gb >= 16:
        return "c", ram_gb
    elif ram_gb >= 8:
        return "b", ram_gb
    else:
        return "a", ram_gb

def main():
    recommended, ram_gb = detect_recommended()
    print("=" * 65)
    print("📚 BAPS Book Translator — Model Downloader")
    print(f"   Detected RAM: ~{ram_gb:.0f} GB")
    print("=" * 65)
    print()
    print("── NLLB-200 (Meta) — Recommended, no token needed ─────────")
    for key, (repo, name, size, ram, note) in NLLB_MODELS.items():
        tag = " ★ RECOMMENDED for your system" if key == recommended else ""
        print(f"  {key}) {name}")
        print(f"     Download: {size}  |  Needs: {ram}  |  {note}{tag}")
    print()
    print("── IndicTrans2 (AI4Bharat) — requires HuggingFace token ───")
    print("  1) indictrans2-indic-indic-1B  (~4.5 GB)  Gujarati/Hindi → Bengali")
    print("  2) indictrans2-en-indic-1B     (~4.5 GB)  English → Bengali")
    print("  3) Both IndicTrans2 models")
    print()
    print("  all) Download NLLB (your recommended tier) + IndicTrans2 indic-indic")
    print("=" * 65)

    choice = input("Enter choice: ").strip().lower()

    # NLLB choices
    if choice in NLLB_MODELS:
        repo, name, _, _, _ = NLLB_MODELS[choice]
        download_nllb_variant(repo, name)
        return

    # IndicTrans2 choices
    if choice in ('1', '2', '3', 'all'):
        token = input("\nHugging Face Token (hf_...): ").strip()
        if not token:
            print("Token required for ai4bharat models.")
            return
        if choice in ('1', '3', 'all'):
            download_indictrans("ai4bharat/indictrans2-indic-indic-1B", token)
        if choice in ('2', '3'):
            download_indictrans("ai4bharat/indictrans2-en-indic-1B", token)
        if choice == 'all':
            repo, name, _, _, _ = NLLB_MODELS[recommended]
            download_nllb_variant(repo, name)
        return

    print("Invalid choice.")

if __name__ == "__main__":
    main()
