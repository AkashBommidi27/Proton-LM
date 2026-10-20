"""
Project Proton: Train Custom BPE Tokenizer on Curated Polymath Corpus
--------------------------------------------------------------------
This script loads our Phase 1 curated polymath dataset and trains a production
custom Byte-Pair Encoding (BPE) tokenizer for Proton.
Target vocabulary size: 4,096 tokens (optimal for Proton-Micro / fast pre-training).
"""

import os
import json
from phase2_tokenizer import ProtonTokenizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "curated", "polymath_curated_sample.jsonl")
TOKENIZER_OUT = os.path.join(os.path.dirname(__file__), "tokenizer_polymath_4k")

def train_polymath_tokenizer(vocab_size=4096, max_chars=1_000_000):
    print("=" * 65)
    print("Project Proton: Training Polymath BPE Tokenizer")
    print(f"Target Vocabulary Size: {vocab_size:,}")
    print("=" * 65)

    assert os.path.exists(DATA_PATH), f"Curated data file not found at: {DATA_PATH}"

    # Load text samples from curated JSONL up to max_chars
    texts = []
    total_chars = 0
    print("Reading text samples from curated dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            doc_text = entry.get("text", "")
            texts.append(doc_text)
            total_chars += len(doc_text)
            if total_chars >= max_chars:
                break

    full_training_text = "\n\n".join(texts)
    print(f"Loaded {len(texts):,} curated documents ({len(full_training_text):,} characters / ~{total_chars/1024:.1f} KB).")

    tokenizer = ProtonTokenizer()
    print("\nStarting iterative BPE training...")
    tokenizer.train(full_training_text, vocab_size=vocab_size, verbose=True)

    # Register standard control & chat special tokens
    special_tokens = [
        "<|bos|>",          # Beginning of sequence
        "<|eos|>",          # End of sequence
        "<|pad|>",          # Padding token
        "<|unk|>",          # Unknown token fallback
        "<|im_start|>",     # Chat turn start
        "<|im_end|>",       # Chat turn end
        "<|system|>",       # System prompt role
        "<|user|>",         # User role
        "<|assistant|>"     # Assistant role
    ]
    tokenizer.register_special_tokens(special_tokens)
    print(f"\nRegistered {len(special_tokens)} special tokens: {special_tokens}")

    # Save to disk
    tokenizer.save(TOKENIZER_OUT)

    # Verification sample
    sample_eval = (
        "<|im_start|>user\n"
        "Explain how the proton was discovered in physics.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The proton was discovered by Ernest Rutherford in 1917.<|im_end|>"
    )
    tokens = tokenizer.encode(sample_eval)
    decoded = tokenizer.decode(tokens)
    print("\n--- Tokenizer Sanity Check ---")
    print(f"Sample length:  {len(sample_eval)} characters")
    print(f"Token count:    {len(tokens)} tokens")
    print(f"Token IDs:      {tokens[:20]} ... (truncated)")
    print(f"Compression:    {len(sample_eval) / len(tokens):.2f} characters per token")
    assert decoded == sample_eval, "Lossless roundtrip check failed!"
    print(">>> PASS: Lossless chat roundtrip verified successfully!")

if __name__ == "__main__":
    train_polymath_tokenizer(vocab_size=1024, max_chars=300_000)
