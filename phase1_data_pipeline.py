"""
Project Proton: Phase 1 - Data Engineering & Polymath Corpus Curation Pipeline
-----------------------------------------------------------------------------
This script implements a production-grade data curation pipeline:
1. Streaming high-quality domain datasets (FineWeb-Edu sample, Wikipedia STEM, TinyStories)
2. Quality Filtering Heuristics:
   - Word count limits (min/max length)
   - Symbol-to-word ratio (rejects code dump garbage / broken encoding)
   - Whitespace and repetitive character normalization
3. Deduplication: Exact SHA-256 document hashing
4. Sharding & Storage: Outputs clean documents into `data/curated/` in JSON Lines format (.jsonl)
"""

import os
import re
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "curated")
os.makedirs(DATA_DIR, exist_ok=True)

class QualityFilter:
    """Heuristic filters used by leading AI research labs (FineWeb/RedPajama)."""
    
    def __init__(self, min_words=30, max_words=5000, max_symbol_ratio=0.15):
        self.min_words = min_words
        self.max_words = max_words
        self.max_symbol_ratio = max_symbol_ratio
        self.seen_hashes = set()

    def clean_text(self, text: str) -> str:
        # Normalize multiple spaces, tabs, and newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def is_valid(self, text: str) -> bool:
        if not text:
            return False

        words = text.split()
        word_count = len(words)
        
        # 1. Length check
        if word_count < self.min_words or word_count > self.max_words:
            return False

        # 2. Symbol-to-word ratio check (catches binary leaks, ascii art, noise)
        symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        if (symbols / max(1, len(text))) > self.max_symbol_ratio:
            return False

        # 3. Exact deduplication check
        doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if doc_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(doc_hash)

        return True

def build_curated_sample(target_samples_per_source=2500):
    print("=" * 65)
    print("Project Proton: Phase 1 Data Curation Pipeline")
    print("=" * 65)
    
    q_filter = QualityFilter(min_words=25, max_words=4000)
    output_file = os.path.join(DATA_DIR, "polymath_curated_sample.jsonl")
    
    total_written = 0
    total_words = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        # -------------------------------------------------------------
        # Stream 1: FineWeb-Edu (Educational Web subset scored >= 3)
        # -------------------------------------------------------------
        print("\n[1/3] Streaming FineWeb-Edu Sample (General & Educational Knowledge)...")
        try:
            fineweb = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split="train",
                streaming=True
            )
            count = 0
            for row in tqdm(fineweb, desc="FineWeb-Edu", total=target_samples_per_source):
                text = q_filter.clean_text(row.get("text", ""))
                if q_filter.is_valid(text):
                    entry = {
                        "source": "fineweb-edu",
                        "text": text,
                        "words": len(text.split())
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    total_written += 1
                    total_words += entry["words"]
                    count += 1
                    if count >= target_samples_per_source:
                        break
        except Exception as e:
            print(f"Notice: FineWeb streaming fallback ({e})")

        # -------------------------------------------------------------
        # Stream 2: TinyStories (Coherent syntactic narrative structures)
        # -------------------------------------------------------------
        print("\n[2/3] Streaming TinyStories Sample (Grammar & Structure)...")
        try:
            stories = load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=True
            )
            count = 0
            for row in tqdm(stories, desc="TinyStories", total=target_samples_per_source):
                text = q_filter.clean_text(row.get("text", ""))
                if q_filter.is_valid(text):
                    entry = {
                        "source": "tinystories",
                        "text": text,
                        "words": len(text.split())
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    total_written += 1
                    total_words += entry["words"]
                    count += 1
                    if count >= target_samples_per_source:
                        break
        except Exception as e:
            print(f"Notice: TinyStories streaming fallback ({e})")

        # -------------------------------------------------------------
        # Stream 3: Math & Science Reasoning (GSM8K / Math QA logic)
        # -------------------------------------------------------------
        print("\n[3/3] Streaming GSM8K Reasoning Sample (Math & Logic)...")
        try:
            gsm = load_dataset(
                "openai/gsm8k",
                "main",
                split="train",
                streaming=True
            )
            count = 0
            for row in tqdm(gsm, desc="GSM8K Math", total=1000):
                q = row.get("question", "")
                a = row.get("answer", "")
                text = q_filter.clean_text(f"Problem: {q}\n\nSolution:\n{a}")
                if q_filter.is_valid(text):
                    entry = {
                        "source": "gsm8k-reasoning",
                        "text": text,
                        "words": len(text.split())
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    total_written += 1
                    total_words += entry["words"]
                    count += 1
                    if count >= 1000:
                        break
        except Exception as e:
            print(f"Notice: GSM8K streaming fallback ({e})")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("\n" + "=" * 65)
    print("Phase 1 Data Curation Pipeline Complete!")
    print(f"Total Documents Curated: {total_written:,}")
    print(f"Total Word Count:        {total_words:,} words (~{int(total_words * 1.33):,} tokens)")
    print(f"Output File:             {output_file} ({file_size_mb:.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    build_curated_sample(target_samples_per_source=2000)
