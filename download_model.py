import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download


def download_indictrans(model_name: str, token: str) -> bool:
    """
    Downloads an IndicTrans2 model via snapshot_download into models/<short-name>/.
    Uses snapshot_download to avoid the tokenizer class resolution issue with older cache formats.
    """
    short_name = model_name.split("/")[-1]
    local_dir = os.path.join("models", short_name)
    print(f"\n--- Downloading: '{model_name}' → {local_dir} ---")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            token=token if token else None,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"✓ Downloaded successfully.")
        # Quick load test
        print("  Verifying tokenizer...")
        AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        print("  ✓ Tokenizer OK")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"  Check: https://huggingface.co/{model_name}")
        print("  You must accept the model terms before downloading.")
        return False


def main():
    print("=" * 65)
    print("📚 BAPS Book Translator — Model Downloader")
    print("=" * 65)
    print()
    print("Select the IndicTrans2 model to download:")
    print()
    print("  1) Indic-to-Indic 1B  ⭐ Recommended")
    print("     For Gujarati / Hindi → Bengali")
    print("     → ai4bharat/indictrans2-indic-indic-1B  (~4.5 GB)")
    print()
    print("  2) English-to-Indic 1B")
    print("     For English → Bengali")
    print("     → ai4bharat/indictrans2-en-indic-1B  (~4.5 GB)")
    print()
    print("  3) Both models")
    print()
    print("=" * 65)
    print("Requires: HuggingFace account + accept model terms at")
    print("  https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B")
    print("=" * 65)

    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    if choice not in ("1", "2", "3"):
        print("Invalid choice.")
        return

    token = input("\nHuggingFace Token (hf_...): ").strip()
    if not token:
        print("Token is required for ai4bharat gated models.")
        return

    success = True
    if choice in ("1", "3"):
        ok = download_indictrans("ai4bharat/indictrans2-indic-indic-1B", token)
        success = success and ok

    if choice in ("2", "3"):
        ok = download_indictrans("ai4bharat/indictrans2-en-indic-1B", token)
        success = success and ok

    if success:
        print("\n🎉 Done! Model files are in 'models/'. Run: streamlit run app.py")
    else:
        print("\n⚠️ One or more downloads failed. Check the errors above.")


if __name__ == "__main__":
    main()
