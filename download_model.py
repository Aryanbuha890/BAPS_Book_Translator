import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download_checkpoint(model_name: str, token: str):
    """
    Downloads and caches a specific Hugging Face model checkpoint.
    """
    print(f"\n--- Downloading: '{model_name}' ---")
    print("Saving tokenizer and weights locally to the 'models/' cache directory...\n")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir="models", 
            token=token
        )
        print(f"✓ Tokenizer for '{model_name}' downloaded successfully.")
        
        # Download model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir="models", 
            token=token
        )
        print(f"✓ Model weights for '{model_name}' downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading '{model_name}': {e}")
        print("Please check:")
        print("1. Your Hugging Face token is correct and active.")
        print(f"2. You have accepted the model terms at: https://huggingface.co/{model_name}")
        return False

def main():
    print("==============================================================")
    print("📚 BAPS Book Translator - Model Downloader")
    print("==============================================================")
    print("Select the IndicTrans2 1B parameter checkpoint to download:")
    print("1. Indic-to-Indic 1B model (For Hindi / Gujarati to Bengali)")
    print("   -> 'ai4bharat/indictrans2-indic-indic-1B'")
    print("2. English-to-Indic 1B model (For English to Bengali)")
    print("   -> 'ai4bharat/indictrans2-en-indic-1B'")
    print("3. Both 1B models (Recommended for full multi-source capability)")
    print("==============================================================")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    if choice not in ('1', '2', '3'):
        print("Error: Invalid choice.")
        return
        
    token = input("\nPlease enter your Hugging Face Token (starts with hf_...): ").strip()
    if not token:
        print("Error: Hugging Face Token is required.")
        return
        
    success = True
    if choice in ('1', '3'):
        ok = download_checkpoint("ai4bharat/indictrans2-indic-indic-1B", token)
        success = success and ok
        
    if choice in ('2', '3'):
        ok = download_checkpoint("ai4bharat/indictrans2-en-indic-1B", token)
        success = success and ok
        
    if success:
        print("\n🎉 Success! All selected model files are cached locally in your 'models/' directory.")
        print("You can now close this window and run the translator fully offline.")
    else:
        print("\n⚠️ Download complete, but one or more operations failed. Please review the error messages above.")

if __name__ == "__main__":
    main()
