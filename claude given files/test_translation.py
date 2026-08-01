"""
test_translation.py

Run this FIRST after downloading the models, before translating a real
document. Confirms: models load, IndicProcessor pre/postprocessing works,
numbers/glossary terms survive translation intact.

Run: python test_translation.py
"""

from translator import translate_chunks, detect_source_lang
from glossary import Glossary
import config

TEST_CASES = [
    ("English", "Bhagwan Swaminarayan visited Gadhada in the year 1801.", config.LANG_ENGLISH),
    ("Hindi", "गुजरात भारत का एक सुंदर राज्य है। यह पुस्तक बहुत रोचक है।", config.LANG_HINDI),
    ("Gujarati", "ગુજરાત ભારતનું એક સુંદર રાજ્ય છે. આ પુસ્તક ખૂબ જ રસપ્રદ છે.", config.LANG_GUJARATI),
]

if __name__ == "__main__":
    glossary = Glossary(config.GLOSSARY_CSV_PATH)

    print("=" * 60)
    print("STEP 1: Language auto-detection check")
    print("=" * 60)
    for label, text, expected in TEST_CASES:
        detected = detect_source_lang(text)
        status = "OK" if detected == expected else "MISMATCH"
        print(f"[{status}] {label}: detected={detected}, expected={expected}")

    print()
    print("=" * 60)
    print("STEP 2: Translation + glossary protection check")
    print("=" * 60)
    for label, text, src_lang in TEST_CASES:
        protected, pmap = glossary.protect(text)
        translated = translate_chunks([protected], src_lang)[0]
        restored = glossary.restore(translated, pmap)

        print(f"\n--- {label} ---")
        print(f"SOURCE:     {text}")
        print(f"BENGALI:    {restored}")
        if "1801" in text:
            number_ok = "1801" in restored
            print(f"Number preserved (1801): {'YES' if number_ok else 'NO -- check IndicProcessor setup'}")

    print("\nIf the Bengali output above looks like real, readable Bengali text")
    print("(not garbled symbols, not English, not empty), your setup is correct")
    print("and you can proceed to translating real documents via `streamlit run app.py`.")
