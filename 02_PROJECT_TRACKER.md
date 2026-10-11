# Project Proton: Progress Tracker & Execution Dashboard

> **Model Name:** **Proton** (Proton-Micro $\to$ Proton-Mini $\to$ Proton-Instruct)  
> **Status:** 🟡 Initiated / Ready to Begin Phase 0  
> **Workspace:** `c:/Users/bommi/Downloads/Proton`  
> **Last Updated:** 2026-10-11

---

## 1. High-Level Progress Overview

| Phase | Description | Estimated Time | Status | Completion % |
|:---:|---|:---:|:---:|:---:|
| **Phase 0** | Mathematical & Engineering Foundations | 1 - 2 weeks | 🟢 Completed | 100% |
| **Phase 1** | Data Engineering: Polymath Mixture (Science, History, Politics, Space, Math, Code) | 1 - 2 weeks | 🟡 Ready to Begin | 0% |
| **Phase 2** | Byte-Pair Encoding (BPE) Tokenizer | 3 - 5 days | ⚪ Not Started | 0% |
| **Phase 3** | Transformer Architecture from Scratch (RoPE, RMSNorm, SwiGLU, GQA) | 2 - 3 weeks | ⚪ Not Started | 0% |
| **Phase 4** | Pre-Training Pipeline (Tier 1 Free Colab $\to$ Tier 2 Cloudrift RTX 4090) | 2 - 3 weeks | ⚪ Not Started | 0% |
| **Phase 5** | Post-Training (SFT & Domain Instruction Tuning) | 1 - 2 weeks | ⚪ Not Started | 0% |
| **Phase 6** | Alignment & Reasoning (DPO / GRPO Reasoning Chains) | 1 - 2 weeks | ⚪ Not Started | 0% |
| **Phase 7** | Evaluation & Standard Benchmarks (PPL, MMLU-Subsets, ARC, GSM8K, HellaSwag) | 1 week | ⚪ Not Started | 0% |
| **Phase 8** | Quantization, GGUF & Local Serving (llama.cpp / Ollama) | 1 week | ⚪ Not Started | 0% |

---

## 2. Granular Task Checklist & Milestones

### Phase 0: Foundations & Environment Setup
- [x] **0.1 Environment Initialization** *(Completed 2026-10-12)*
  - [x] Check Python version (Python 3.11 Windows AMD64)
  - [x] Set up virtual environment (`.venv`)
  - [x] Install PyTorch with CPU mode locally (`torch 2.14.1`, `numpy 2.4.6`, `matplotlib 3.11.2`)
  - [x] Verify hardware & GPU availability plan (Cloud RTX 4090 for Phase 4)
- [x] **0.2 Math & Tensor Intuition Exercises** *(Completed 2026-10-12)*
  - [x] Understand Tensor dimensions: Shapes, Strides, Broadcasting
  - [x] Implement Matrix Multiplication by hand vs. `torch.matmul`
  - [x] Implement Softmax and numerically stable Softmax (`x - max(x)`)
  - [x] Implement Cross-Entropy Loss from raw probabilities (`-ln(P_target)`)
  - [x] Script verified: [`phase0_foundations.py`](file:///c:/Users/bommi/Downloads/Proton/phase0_foundations.py)
- [x] **0.3 The Simplest Neural Network** *(Completed 2026-10-12)*
  - [x] Build a 2-layer Character-level Bigram / MLP model
  - [x] Train it to overfit on corpus to verify backpropagation mechanics
  - [x] Connect MLflow to track loss, perplexity, hyperparameters, and artifacts
  - [x] Script verified: [`phase0_simplest_lm.py`](file:///c:/Users/bommi/Downloads/Proton/phase0_simplest_lm.py) (Loss dropped from 3.20 $\to$ 0.016, PPL dropped to 1.02, generated coherent text completions)
  - [x] Checkpoint saved: `checkpoints/phase0_char_mlp.pt`

---

### Phase 1: Data Engineering & Preprocessing
- [ ] **1.1 Dataset Sourcing**
  - [ ] Download raw sample corpus (e.g., TinyStories, Wikipedia excerpt, or FineWeb-Edu sample)
  - [ ] Understand target token counts (e.g., 50M tokens for toy, 1B tokens for mini)
- [ ] **1.2 Data Cleaning & Filtering**
  - [ ] Implement length and whitespace normalization
  - [ ] Implement language/quality heuristics
  - [ ] Implement basic deduplication (exact hash / MinHash)
- [ ] **1.3 Data Packing & Sharding**
  - [ ] Save dataset into memory-mapped raw binary format (`.bin` / uint16 or uint32) for fast zero-copy loading during training

---

### Phase 2: Tokenization From Scratch
- [ ] **2.1 Tokenizer Theory & Setup**
  - [ ] Understand UTF-8 byte encodings vs. Unicode characters
  - [ ] Byte-level representation ($0 - 255$)
- [ ] **2.2 BPE Training Algorithm**
  - [ ] Implement pair frequency counting
  - [ ] Implement iterative merge operations
  - [ ] Train vocabulary of size $V = 4,096$ (toy) or $V = 32,000$ (full)
- [ ] **2.3 Tokenizer Class Implementation**
  - [ ] Implement `encode(text)` $\to$ `List[int]`
  - [ ] Implement `decode(tokens)` $\to$ `str`
  - [ ] Add special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`)
  - [ ] Save and load custom tokenizer files (`vocab.json`, `merges.txt`)

---

### Phase 3: Transformer Architecture from First Principles
- [ ] **3.1 Embedding Layer**
  - [ ] Token embedding lookup table
- [ ] **3.2 Rotary Positional Embeddings (RoPE)**
  - [ ] Implement frequency calculation ($\theta_i = 10000^{-2(i-1)/d}$)
  - [ ] Apply complex rotation matrix to Query and Key vectors
- [ ] **3.3 Attention Mechanism**
  - [ ] Implement Scaled Dot-Product Attention with causal triangular mask
  - [ ] Implement Multi-Head Attention (MHA)
  - [ ] Upgrade to Grouped-Query Attention (GQA) for memory savings
- [ ] **3.4 Normalization & Activations**
  - [ ] Implement RMSNorm (Root Mean Square Layer Normalization)
  - [ ] Implement SwiGLU feed-forward layer (`Swish(xW) * xV * W_out`)
- [ ] **3.5 Complete Transformer Block & Model**
  - [ ] Assemble Pre-RMSNorm residual connections
  - [ ] Assemble `DecoderBlock` $\to$ Stack $N$ layers
  - [ ] Add final RMSNorm and Linear LM Head
  - [ ] Implement KV-Cache for fast generation

---

### Phase 4: Pre-Training Infrastructure & Scaled Training
- [ ] **4.1 Training Pipeline Foundations**
  - [ ] Implement efficient `DataLoader` streaming batches from disk
  - [ ] Configure AdamW optimizer with weight decay on 2D weights only
  - [ ] Implement Cosine Learning Rate Schedule with Warmup
  - [ ] Implement Gradient Clipping (`torch.nn.utils.clip_grad_norm_`)
- [ ] **4.2 Micro-Pretraining Run (Tier 1: 15M Params)**
  - [ ] Model architecture: 6 layers, 6 heads, $d_{model} = 384$, context = 256
  - [ ] Train on TinyStories or Shakespeare (Cost: $0)
  - [ ] Verify validation loss drops below ~1.5
  - [ ] Generate sample text completions to confirm coherence
- [ ] **4.3 Scaled Pretraining Run (Tier 2: 100M Params - Optional Cloud Run)**
  - [ ] Rent GPU (RTX 4090 on RunPod or Vast.ai)
  - [ ] Train on 1B tokens using mixed-precision BF16
  - [ ] Save model checkpoints (`checkpoint_step_XXXX.pt`)

---

### Phase 5: Post-Training (SFT / Instruction Tuning)
- [ ] **5.1 Instruction Dataset Preparation**
  - [ ] Prepare prompt-response pairs (Alpaca or OpenHermes format)
  - [ ] Define Chat Template with delimiting tokens (`<|im_start|>user...`)
- [ ] **5.2 SFT Training Engine**
  - [ ] Implement response-only loss masking (masking prompt tokens with `-100`)
  - [ ] Fine-tune base model weights
  - [ ] Implement LoRA (Low-Rank Adaptation) adapter training
- [ ] **5.3 Verification**
  - [ ] Test chatbot responses to multi-turn conversational queries

---

### Phase 6: Alignment & Reasoning (DPO & GRPO)
- [ ] **6.1 Preference Data & DPO**
  - [ ] Collect chosen vs. rejected pairs
  - [ ] Implement Direct Preference Optimization (DPO) loss function
  - [ ] Train policy model vs. reference model
- [ ] **6.2 Reasoning & Verification (DeepSeek-R1 / GRPO Style)**
  - [ ] Implement rule-based reward function (math answers / code execution)
  - [ ] Encourage `<think>` ... `</think>` chain-of-thought generation

---

### Phase 7: Evaluation & Benchmarking
- [ ] **7.1 Perplexity Benchmark**
  - [ ] Calculate test set cross-entropy loss and Perplexity score
- [ ] **7.2 Zero-Shot Task Evaluation**
  - [ ] Test on GSM8K (Math) or ARC-Easy (Reasoning)
  - [ ] Run benchmark with `lm-evaluation-harness`

---

### Phase 8: Quantization & Deployment
- [ ] **8.1 Model Conversion**
  - [ ] Export PyTorch weights to `Safetensors`
  - [ ] Convert model to GGUF format
- [ ] **8.2 Local Serving**
  - [ ] Run 4-bit / 8-bit quantized model using `llama.cpp`
  - [ ] Integrate with Ollama or local web UI (OpenWebUI)

---

## 3. Hardware, Compute & Cost Log

| Date | Task / Run Name | Model Size | Hardware Used | Runtime | Total Cost ($) | Result / Checkpoint |
|---|---|---|---|---|---|---|
| *Pending* | Phase 0 Setup | N/A | Local PC (Windows) | - | $0.00 | Environment verified |
| *Pending* | Tier 1 Micro-Pretraining | ~15M | Local CPU / GPU or Free Colab | ~1 hr | $0.00 | First generated text |

---

## 4. How to Use & Maintain This Tracker
1. **At the start of every session:** Review the active milestone.
2. **When code is completed:** Mark the checkbox with `[x]`, record the completion date, and link the script file.
3. **When a training run finishes:** Log the runtime, cost, loss value, and checkpoint in Section 3.
