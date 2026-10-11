"""
Project Proton: Phase 0.2 - Mathematical & Tensor Intuition
-----------------------------------------------------------
This script demystifies the core operations of LLMs:
1. Tensor Shapes, Strides & Broadcasting
2. Matrix Multiplication (Manual loops vs. torch.matmul)
3. Logits to Probabilities: Numerically Stable Softmax
4. Categorical Cross-Entropy Loss from First Principles
"""

import math
import torch
import numpy as np

print("=" * 60)
print("Project Proton: Phase 0.2 Foundations")
print("=" * 60)

# -------------------------------------------------------------
# Part 1: Tensor Shapes & Broadcasting Intuition
# -------------------------------------------------------------
print("\n[1] TENSOR BROADCASTING")
# In an LLM, we often add a 1D bias vector or positional vector (e.g. shape [D])
# to a 3D batch representation (shape [Batch, Sequence_Len, D]).
batch_tokens = torch.randn(2, 4, 8)  # Batch=2, SeqLen=4, EmbeddingDim=8
bias = torch.randn(8)                # Dim=8

# PyTorch broadcasts `bias` automatically across Batch and SeqLen dimensions:
result = batch_tokens + bias
print(f"Batch shape: {batch_tokens.shape} + Bias shape: {bias.shape} -> Result shape: {result.shape}")


# -------------------------------------------------------------
# Part 2: Matrix Multiplication (The core of Q * K^T and Feed-Forward)
# -------------------------------------------------------------
print("\n[2] MATRIX MULTIPLICATION FROM FIRST PRINCIPLES")
# Let A be (M, K) and B be (K, N). The output C is (M, N).
A = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], 
                  [7.0, 8.0]])

# Manual implementation using raw nested loops (O(M * N * K)):
M, K_a = A.shape
K_b, N = B.shape
assert K_a == K_b, "Inner dimensions must match!"

C_manual = torch.zeros(M, N)
for i in range(M):
    for j in range(N):
        total = 0.0
        for k in range(K_a):
            total += A[i, k] * B[k, j]
        C_manual[i, j] = total

C_pytorch = torch.matmul(A, B)
print("Manual Matmul Output:\n", C_manual)
print("PyTorch Matmul Output:\n", C_pytorch)
assert torch.allclose(C_manual, C_pytorch), "Manual matmul does not match PyTorch!"
print(">>> Verification: Manual Matmul matches PyTorch exactly.")


# -------------------------------------------------------------
# Part 3: Softmax - Why "Naive Softmax" Breaks & Stable Softmax
# -------------------------------------------------------------
print("\n[3] NUMERICALLY STABLE SOFTMAX")
# When an LLM outputs raw logits, they can be large numbers like 100.0 or 1000.0.
# e^1000 overflows into 'inf' (NaN) in float arithmetic!
large_logits = torch.tensor([1000.0, 1001.0, 1002.0])

def naive_softmax(x):
    # This will fail on large numbers: e^1000 = inf, inf / inf = NaN!
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x)

def stable_softmax(x, dim=-1):
    # Mathematical trick: Softmax(x) == Softmax(x - max(x))
    # Proof: e^(x - c) / sum(e^(x - c)) = (e^x * e^-c) / (e^-c * sum(e^x)) = e^x / sum(e^x)
    # Subtracting the maximum ensures the highest exponent is e^0 = 1.0 (never overflows!).
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(x - max_val)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)

print("Naive Softmax with large logits:", naive_softmax(large_logits))  # [nan, nan, nan]
print("Stable Softmax with large logits:", stable_softmax(large_logits)) # Clean probabilities!
print("PyTorch official Softmax:       ", torch.softmax(large_logits, dim=-1))


# -------------------------------------------------------------
# Part 4: Cross-Entropy Loss from First Principles
# -------------------------------------------------------------
print("\n[4] CROSS-ENTROPY LOSS FROM SCRATCH")
# Suppose the vocabulary has 5 tokens: [0, 1, 2, 3, 4]
# The model outputs logits for 1 position:
logits = torch.tensor([2.0, 1.0, 0.1, 0.05, 3.5])
# The true target token that should have been predicted is token ID 4:
target_token = torch.tensor(4)

# 1. Convert logits to probabilities:
probs = stable_softmax(logits)
# 2. Extract probability assigned to the target token:
target_prob = probs[target_token]
# 3. Cross-Entropy Loss is negative log-likelihood: -ln(P_target)
loss_manual = -torch.log(target_prob)

# PyTorch built-in:
loss_fn = torch.nn.CrossEntropyLoss()
loss_pytorch = loss_fn(logits.unsqueeze(0), target_token.unsqueeze(0))

print(f"Target token: {target_token.item()}")
print(f"Assigned probability: {target_prob.item():.4f} ({target_prob.item() * 100:.2f}%)")
print(f"Manual Cross-Entropy Loss:  {loss_manual.item():.6f}")
print(f"PyTorch CrossEntropyLoss:   {loss_pytorch.item():.6f}")
assert torch.isclose(loss_manual, loss_pytorch), "Losses do not match!"
print(">>> Verification: Manual Cross-Entropy matches PyTorch exactly.")

print("\n" + "=" * 60)
print("Phase 0.2 Foundation Checks Passed Successfully!")
print("=" * 60)
