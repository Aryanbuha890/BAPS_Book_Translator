import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download():
    model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
    token = input("Please enter your Hugging Face Token (starts with hf_...): ").strip()
    
    if not token:
        print("Error: Hugging Face Token is required.")
        return
        
    print(f"\nDownloading tokenizer and model weights for '{model_name}'...")
    print("Saving to local 'models/' directory inside your project folder.\n")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir="models", 
            token=token
        )
        print("✓ Tokenizer downloaded successfully.")
        
        # Download model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir="models", 
            token=token
        )
        print("✓ Model weights downloaded successfully.")
        print("\n🎉 Success! All model files are saved offline. You can now close this and start translating.")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Please check that your Hugging Face token is correct and you have accepted the model terms at:")
        print("https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M")

if __name__ == "__main__":
    download()
