# Project Proton: Master Knowledge Base & Learning Notes

> **Model:** **Proton** (A Custom LLM Built From Scratch)  
> **Purpose:** A living reference notebook containing every concept, intuition, mathematical derivation, architectural choice, and discussion log as we build an LLM from scratch.  
> **Workspace:** `c:/Users/bommi/Downloads/Proton`  
> **Status:** Active & Expanding

---

## Table of Contents
1. [Core Mental Models: What Truly is an LLM?](#1-core-mental-models-what-truly-is-an-llm)
2. [Glossary of Essential Terms (Plain English)](#2-glossary-of-essential-terms-plain-english)
3. [The Mathematics Demystified (Zero Jargon)](#3-the-mathematics-demystified-zero-jargon)
4. [The Transformer Mechanism from First Principles](#4-the-transformer-mechanism-from-first-principles)
5. [Hardware, VRAM & Compute Mechanics](#5-hardware-vram--compute-mechanics)
6. [Why Leading Labs Make Specific Architectural Choices](#6-why-leading-labs-make-specific-architectural-choices)
7. [Questions & Discussions Log (Interactive Journal)](#7-questions--discussions-log-interactive-journal)

---

## 1. Core Mental Models: What Truly is an LLM?

### 1.1 The Ultimate Next-Token Predictor
At its core, a Large Language Model is simply a **probabilistic function**:
$$P(\text{next token} \mid \text{previous tokens})$$

If you give it the sequence:
> `"The quick brown fox jumps over the lazy ..."`

The model does not "think" or "feel". It computes a probability score for every single word in its dictionary (vocabulary):
- `"dog"`: $82.4\%$
- `"cat"`: $8.1\%$
- `"fox"`: $1.2\%$
- `"banana"`: $0.00001\%$

The entire complexity of reasoning, writing code, translating languages, and chatting is an **emergent property** of learning how to predict the next token accurately across trillions of diverse texts.

### 1.2 The Two Halves of an LLM: Weights vs. Code
- **The Model Architecture (The Code):** A fixed mathematical blueprint (~300 lines of PyTorch code) describing how numbers are multiplied, added, and rotated.
- **The Parameters / Weights (The File):** A giant file of decimal numbers (floats). A 1B model has 1,000,000,000 numbers. Training an LLM is simply nudging these numbers slightly so the next token predictions become less wrong.

---

## 2. Glossary of Essential Terms (Plain English)

| Term | Simple Definition | Concrete Example / Analogy |
|---|---|---|
| **Token** | A chunk of text (a character, syllable, or subword) mapped to an integer ID. | The word `"learning"` might be split into token `1532` (`"learn"`) and token `278` (`"ing"`). |
| **Vocabulary ($V$)** | The dictionary of all tokens the model understands. | GPT-4 has ~$100,000$ tokens; Llama 3 has $128,256$ tokens. |
| **Embedding** | Converting a token integer into a high-dimensional vector of numbers. | Token `1532` $\to$ a vector of 768 decimal numbers capturing its semantic meaning. |
| **Context Window ($T$)** | The maximum number of tokens the model can look back at simultaneously. | Llama-3 has 8k (standard) or 128k tokens; GPT-4o has 128k tokens. |
| **Parameters ($N$)** | The tunable weight matrices inside the neural network. | A "7B" model has 7 billion learnable numbers. |
| **Loss** | A single number indicating how wrong the model was on its prediction. | 0.0 is perfect prediction; high loss means the model was guessing randomly. |
| **Perplexity (PPL)** | $e^{\text{Loss}}$, representing the effective number of choices the model is confused between. | If PPL = 10, the model is guessing between roughly 10 words. |
| **Epoch** | One complete pass through the entire training dataset. | For LLM pre-training, we usually do **1 single epoch** over trillions of tokens! |

---

## 3. The Mathematics Demystified (Zero Jargon)

### 3.1 Vectors and Dot Products: The Engine of "Similarity"
A **vector** is just a list of numbers representing coordinates in space:
$$\vec{A} = [0.8, -0.2, 0.5]$$
$$\vec{B} = [0.7, -0.1, 0.6]$$

The **dot product** multiplies corresponding numbers and sums them:
$$\vec{A} \cdot \vec{B} = (0.8 \times 0.7) + (-0.2 \times -0.1) + (0.5 \times 0.6) = 0.56 + 0.02 + 0.30 = 0.88$$

- If two vectors point in the exact same direction, the dot product is high ($\to$ **high semantic similarity**).
- If they are orthogonal (perpendicular), the dot product is $0$ ($\to$ **unrelated**).
- This is how the model mathematically measures: *"Is this word related to that word?"*

### 3.2 Softmax: Turning Raw Scores into Percentages
Neural networks output raw, unconstrained real numbers called **logits** (e.g., `[2.5, 0.1, -1.8]`).
Softmax converts them into clean probabilities that:
1. Are all positive ($> 0$).
2. Sum up exactly to $1.0$ ($100\%$).

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### 3.3 Cross-Entropy Loss: Measuring the Error
If the correct next word was `"apple"`, and the model assigned `"apple"` a probability of $P$:
$$\text{Loss} = -\ln(P)$$
- If the model was $100\%$ sure ($P=1.0$), Loss $= -\ln(1.0) = 0$ (Zero penalty!).
- If the model was only $1\%$ sure ($P=0.01$), Loss $= -\ln(0.01) = 4.60$ (Huge penalty!).

---

## 4. The Transformer Mechanism from First Principles

### 4.1 The Attention Mechanism (Query, Key, Value)
Think of Attention like searching in a library or database:
1. **Query ($Q$):** What the current token is looking for.  
   *(e.g., The pronoun `"it"` is asking: "What noun was just mentioned?")*
2. **Key ($K$):** The label or index of every preceding token.  
   *(e.g., The word `"cat"` has a Key that says: "I am a singular animal noun.")*
3. **Value ($V$):** The actual informational content that gets retrieved.

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

- $Q K^T$: Matches queries with keys to produce an **attention score matrix**.
- $\frac{1}{\sqrt{d_k}}$: Scales down the numbers so gradients don't vanish.
- $\text{Mask}$: Ensures tokens can only look into the past (a token cannot cheat and look at future words!).
- $\text{Softmax}(\dots)$: Produces attention weights (percentages).
- $\times V$: Aggregates the values weighted by attention.

### 4.2 Grouped-Query Attention (GQA)
In traditional Multi-Head Attention (MHA), every query head has its own Key and Value heads (e.g., 32 Q heads, 32 K heads, 32 V heads).
- Problem: Storing all those K and V vectors for long documents eats dozens of gigabytes of GPU VRAM (the KV-cache problem).
- Solution in Llama 3 & Mistral (GQA): 32 Query heads share only 8 Key/Value heads.
- Result: **4x reduction in KV-cache memory** with virtually zero drop in accuracy!

### 4.3 Rotary Position Embeddings (RoPE)
Original transformers added fixed position numbers to word embeddings ($E + P$).
- RoPE's genius: Instead of *adding* positions, it rotates the Query and Key vectors in 2D slices by an angle proportional to the token's position:
$$\vec{q}_m = \mathbf{R}_m \vec{q}$$
- When computing $Q_m K_n^T$, the result depends only on the **relative distance** $(m - n)$ between the words, allowing models to generalize to long sequences effortlessly.

---

## 5. Hardware, VRAM & Compute Mechanics

### 5.1 Why GPUs Instead of CPUs?
- A modern CPU has 8 to 32 high-performance cores. Great for complex sequential logic.
- A modern GPU (e.g., NVIDIA RTX 4090) has **16,384 CUDA cores** and **512 Tensor Cores**.
- Matrix multiplication consists of millions of independent multiplications: GPUs run all of them in parallel in a single clock cycle.

### 5.2 Number Formats (Precision)
- **FP32 (Standard Float):** 32 bits (4 bytes). High precision, but wastes memory and runs slower.
- **FP16 (Half Float):** 16 bits (2 bytes). Fast, but small exponent range causes underflow/overflow (training crashes).
- **BF16 (Brain Float 16):** 16 bits (2 bytes). Has the exact same dynamic range as FP32, but with 16-bit speed. **The universal gold standard for training modern LLMs.**
- **INT8 / INT4:** Used primarily for inference quantization (slashing memory by 50%-75%).

### 5.3 FlashAttention
Why was training LLMs slow before 2022?
- GPUs have two memory types:
  - **HBM (High Bandwidth Memory):** Big (24GB - 80GB), but relatively slow to read/write.
  - **SRAM:** Tiny (a few megabytes per streaming multiprocessor), but blazing fast.
- Standard attention wrote the giant $N \times N$ attention matrix to HBM repeatedly.
- FlashAttention breaks the calculation into tiles, computes attention in SRAM, and never writes the huge intermediate matrix to slow HBM. **Result: 2x - 4x speedup!**

---

## 6. Why Leading Labs Make Specific Architectural Choices

| Component | What Llama 3 / DeepSeek / Mistral Use | The Technical Justification |
|---|---|---|
| **RMSNorm** | Root Mean Square Normalization | Removes mean calculation from standard LayerNorm; saves memory traffic and is 10-20% faster. |
| **SwiGLU** | Gated Swish activation in FFN | Uses 3 weight matrices ($W_{gate}, W_{up}, W_{down}$). Consistently achieves lower cross-entropy loss than GELU. |
| **Untied Embeddings** | Separate Input & Output Matrices | Older models tied input and output token matrices. Modern large models untie them to allow richer language representations. |
| **DPO over RLHF** | Direct Preference Optimization | Eliminates the unstable PPO actor-critic loop; stable closed-form optimization directly on text pairs. |

---

## 7. Questions & Discussions Log (Interactive Journal)

*(This section will be continuously updated after every discussion and question.)*

### Entry 001: Project Launch & First Principles (2026-10-11)
- **Question:** How does a beginner with zero prior touch build an LLM from scratch?
- **Key Realization:** An LLM is not built in one monolithic leap. It is built in 8 well-defined phases:
  1. Foundations (Python, PyTorch, Math)
  2. Data Collection & Cleaning
  3. Tokenization (BPE)
  4. Transformer Architecture (PyTorch coding)
  5. Pre-Training (Next-token prediction)
  6. SFT (Instruction following)
  7. Alignment & Reasoning (DPO / GRPO)
  8. Quantization & Deployment (GGUF / Ollama)
- **Budget Reality:** A fully functional educational foundation model can be built for **$0** (free Colab/laptop), and a capable small SLM can be pre-trained on rented cloud GPUs for **$5 to $15**.


### Entry 003: Model Sizing Strategy & Domain Topic Selection (2026-10-11)
- **Available Hardware & Budget:**
  - Free tiers: Kaggle (2x T4 or P100 GPUs, 30 hrs/week free), Google Colab (T4 free).
  - Paid GPU credit: \$10 credit on Cloudrift (can rent an RTX 4090 24GB VRAM at ~$0.40-$0.50/hr $\to$ ~20-25 hours of compute).
- **Recommended 2-Stage Sizing Strategy:**
  1. **Stage 1 (Proton-Micro, ~15M - 20M params):**
     - Target: Master every component, verify loss convergence ($<1.8$), bug-free tokenizer & model code.
     - Compute: 100% Free on Kaggle/Colab T4 (runs in ~1-2 hours, \$0 cost).
  2. **Stage 2 (Proton-Mini, ~80M - 120M params):**
     - Target: Solid foundation model with genuine coherence, syntax correctness, and knowledge retention.
     - Compute: Cloudrift RTX 4090 (24GB VRAM) using mixed precision BF16 + FlashAttention/SDPA.
     - Duration & Cost: ~8-14 hours on ~1.5B - 2.5B tokens ($\approx \$4 - \$7$ spent, leaving credit cushion).
- **Domain Topic Candidates & Recipes:**
  1. *The Polymath / Curated Renaissance Model (Recommended for broad curiosity):*
     - Mixture: 35% Science & AI (arXiv/Wikipedia STEM), 25% History, Politics & Geography, 20% Math & Logic, 20% Cleaned Code/Python.
     - Dataset backbone: **FineWeb-Edu** (filtered by topic classifiers) + Wikipedia + Cosmo-corpus + TinyGSM.
  2. *The Cosmos & Earth Explorer (Astrophysics, Space, Geography & Geopolitics):*
     - NASA open archives, planetary science papers, world history & geopolitical compendiums.
  3. *The STEM & AI Engine (Science, Math, AI Research & Code):*
     - Dense arXiv distillations, AI lecture notes, Python implementations, and multi-step derivations.
  4. *Python Coding & System Assistant:*
     - Code generation, docstring completion, algorithmic debugging.
- **Hardware Architecture Comparison: V100 SXM vs. RTX 4090:**
  - *V100 SXM2 (Volta, 2017):* 16GB or 32GB HBM2 (high memory bandwidth ~900 GB/s), but **lacks native BF16** (only FP16, risk of loss underflow) and **cannot run FlashAttention-2** (requires Ampere+ sm_80).
  - *RTX 4090 (Ada Lovelace, 2022):* 24GB GDDR6X, **full native BF16 support**, modern Tensor Cores with FP8/TF32, full **FlashAttention-2 & PyTorch SDPA** support, and nearly 3x raw FP16/BF16 compute TFLOPS.

### Entry 004: Production MLOps Stack & Tooling Architecture (2026-10-12)

#### 1. MLflow: The Experiment & Model Lifecycle Engine
- **What is MLflow?** An open-source platform created by Databricks for tracking machine learning experiments, logging metrics, packaging code, and registering models.
- **Why we use it in Project Proton:**
  - Training an LLM takes hours or days. You cannot rely on raw print statements in a terminal.
  - MLflow automatically creates a local dashboard (`mlflow ui`) where you can see:
    - **Hyperparameters:** Learning rate ($3 \times 10^{-4}$), batch size (64), context length (512), warmup steps (2000).
    - **Live Metric Curves:** Step-by-step train loss, validation loss, perplexity, and gradient norm over time.
    - **Artifacts:** Checkpoint files (`model_step_10000.pt`), tokenizer vocabularies, and sample generated text completions.
    - **Model Comparison:** When you train with different learning rates or architectures (e.g. RMSNorm vs LayerNorm), MLflow plots both runs on the exact same graph so you instantly see which converged faster.

#### 2. Apache Airflow: The Workflow & Pipeline Orchestrator
- **What is Apache Airflow?** A workflow management platform created by Airbnb to programmatically author, schedule, and monitor data pipelines using Directed Acyclic Graphs (DAGs) in Python.
- **Why and when is Airflow used in LLM Engineering?**
  - **In Frontier Labs:** Enterprise LLMs are continuously retrained or updated. Airflow runs daily/weekly automated pipelines:
    - *Task 1:* Scrape fresh academic papers or web dumps at midnight.
    - *Task 2:* Run deduplication and MinHash filtering.
    - *Task 3:* Tokenize the clean text into binary shards.
    - *Task 4:* Trigger an evaluation or fine-tuning job on a GPU cluster.
  - **Operating Nature:** Airflow is a full distributed orchestrator designed natively for Linux environments (it requires a background scheduler daemon and a metadata database like PostgreSQL). In production, it runs inside Docker containers or on Linux cloud instances.
- **How we integrate both in Project Proton:**
  - **MLflow:** Used directly in our training scripts to track training metrics, loss, and checkpoints with a visual web dashboard.
  - **Pipeline Orchestration:** We will structure our Phase 1 data processing as clean, modular tasks (DAGs) that can be run standalone or orchestrated with Airflow when deploying to our cloud environment.

