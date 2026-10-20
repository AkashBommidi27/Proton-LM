"""
Project Proton: Phase 2 - Byte-Pair Encoding (BPE) Tokenizer from Scratch
-------------------------------------------------------------------------
This module builds a standalone byte-level BPE tokenizer (inspired by GPT-2, minbpe, & tiktoken).
Key concepts implemented:
1. UTF-8 raw byte representation (base alphabet of 256 byte tokens: 0..255).
   - Guarantees 0% out-of-vocabulary (<UNK>) words across any language or unicode symbol!
2. Pair frequency counting across raw byte sequences.
3. Iterative pairwise merges to learn new subword tokens.
4. Fast encoding with learned merge rules.
5. Decoding bytes back into valid UTF-8 strings.
6. Special tokens management (<|bos|>, <|eos|>, <|pad|>, <|im_start|>, <|im_end|>).
7. Serialization & Deserialization: save and load vocabulary and merge tables to disk.
"""

import os
import json
from typing import List, Dict, Tuple

class ProtonTokenizer:
    """A First-Principles Byte-Level Byte-Pair Encoding (BPE) Tokenizer."""

    def __init__(self):
        # 1. Base vocabulary: 256 initial byte tokens (0..255)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Merges map: (token_id_1, token_id_2) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Special tokens: mapped to IDs beyond the vocab range
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}

    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Counts frequencies of consecutive token pairs."""
        counts = {}
        for p0, p1 in zip(ids, ids[1:]):
            pair = (p0, p1)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replaces all occurrences of `pair` with `new_id` in the token list."""
        new_ids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int = 1024, verbose: bool = True):
        """Trains BPE on raw text up to the specified `vocab_size`."""
        assert vocab_size >= 256, "Vocab size must be at least 256 for base bytes."
        num_merges = vocab_size - 256
        
        # Convert text into raw UTF-8 byte integers (0..255)
        raw_bytes = list(text.encode("utf-8"))
        ids = list(raw_bytes)

        if verbose:
            print(f"Initial raw byte count: {len(ids):,}")
            print(f"Target vocabulary size: {vocab_size:,} ({num_merges} merges)")

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # No pairs with frequency >= 2 left to merge
                break

            new_id = 256 + i
            ids = self.merge_tokens(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            
            # Combine bytes of the two constituent tokens
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and ((i + 1) % 50 == 0 or (i + 1) == num_merges or (i + 1) <= 5):
                decoded_preview = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"Merge {i+1:4d}/{num_merges}: {best_pair} -> {new_id} ({decoded_preview!r}, freq: {stats[best_pair]})")

        compression_ratio = len(raw_bytes) / max(1, len(ids))
        if verbose:
            print(f"\nTraining Complete!")
            print(f"Final Vocab Size: {len(self.vocab):,}")
            print(f"Compressed from {len(raw_bytes):,} bytes to {len(ids):,} tokens.")
            print(f"Compression Ratio: {compression_ratio:.2f}x (characters per token: {compression_ratio:.2f})")

    def register_special_tokens(self, special_tokens: List[str]):
        """Adds custom control tokens (e.g. <|bos|>, <|eos|>) after the learned vocab."""
        start_id = max(self.vocab.keys(), default=255) + 1
        for i, token_str in enumerate(special_tokens):
            token_id = start_id + i
            self.special_tokens[token_str] = token_id
            self.inverse_special_tokens[token_id] = token_str

    def encode(self, text: str) -> List[int]:
        """Encodes text into a list of token IDs using learned merge rules."""
        # Handle special tokens first if present
        for special_str, special_id in self.special_tokens.items():
            if special_str in text:
                # Split around special token
                parts = text.split(special_str)
                encoded = []
                for idx, part in enumerate(parts):
                    if part:
                        encoded.extend(self.encode(part))
                    if idx < len(parts) - 1:
                        encoded.append(special_id)
                return encoded

        # Convert to raw byte sequence
        ids = list(text.encode("utf-8"))
        
        # Repeatedly apply known merges in the exact order they were learned
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            # Find the pair with the lowest merge index (earliest learned)
            pair_to_merge = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair_to_merge not in self.merges:
                break
            ids = self.merge_tokens(ids, pair_to_merge, self.merges[pair_to_merge])

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back into a human-readable UTF-8 string."""
        byte_chunks = []
        for token_id in ids:
            if token_id in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append(b"")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, output_dir: str):
        """Saves tokenizer vocabulary and merges to disk."""
        os.makedirs(output_dir, exist_ok=True)
        # Convert merges tuple keys to strings for JSON
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        # Convert byte vocab to latin-1 or hex strings
        vocab_serializable = {k: v.decode("latin-1") for k, v in self.vocab.items()}
        
        with open(os.path.join(output_dir, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(merges_serializable, f, indent=2)
            
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2)

        with open(os.path.join(output_dir, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, indent=2)
            
        print(f"Tokenizer saved successfully to: {output_dir}")

    def load(self, model_dir: str):
        """Loads tokenizer state from disk."""
        with open(os.path.join(model_dir, "merges.json"), "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
            self.merges = {tuple(map(int, k.split(","))): v for k, v in raw_merges.items()}

        with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            self.vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}

        with open(os.path.join(model_dir, "special_tokens.json"), "r", encoding="utf-8") as f:
            self.special_tokens = json.load(f)
            self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

if __name__ == "__main__":
    print("=" * 65)
    print("Project Proton: Phase 2 - Tokenizer Self-Test")
    print("=" * 65)
    
    # 1. Quick Unit Test on Small Text
    sample_text = (
        "The proton is a subatomic particle. "
        "Protons and neutrons form the nucleus of an atom. "
        "Language models predict the next token in a sequence."
    )
    
    tokenizer = ProtonTokenizer()
    tokenizer.train(sample_text, vocab_size=280, verbose=True)
    
    # Register special tokens
    special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]
    tokenizer.register_special_tokens(special_tokens)
    
    # Test Encoding & Decoding Round-Trip
    test_sentence = "<|bos|> The proton particle forms the atom! <|eos|>"
    encoded_ids = tokenizer.encode(test_sentence)
    decoded_text = tokenizer.decode(encoded_ids)
    
    print("\n--- Encoding Verification ---")
    print("Original Text:", test_sentence)
    print("Token IDs:    ", encoded_ids)
    print("Decoded Text: ", decoded_text)
    assert decoded_text == test_sentence, "Decoding did not match original text!"
    print(">>> PASS: Lossless roundtrip verified!")

    # Save to disk
    save_path = os.path.join(os.path.dirname(__file__), "tokenizer_test")
    tokenizer.save(save_path)
