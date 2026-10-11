"""
Project Proton: Phase 0.3 - The Simplest Neural Network (Character-level MLP LM)
-------------------------------------------------------------------------------
This script builds and trains a minimal Neural Language Model from first principles:
1. Custom Character Tokenizer (string <-> integer mapping)
2. Dataset formatting: sliding context window -> next-character target
3. Embedding Layer + Hidden Layer with Tanh activation + Linear Output Head
4. Full Training Loop with PyTorch AdamW & Cross-Entropy Loss
5. MLflow integration: Tracking hyperparameters, loss curves, and model artifacts
6. Character generation: Sampling text completions from the trained model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

# -------------------------------------------------------------
# 1. Prepare Toy Corpus & Tokenizer
# -------------------------------------------------------------
corpus = (
    "the proton is a subatomic particle with a positive electric charge. "
    "protons and neutrons make up the nucleus of an atom. "
    "learning to build large language models from scratch starts with understanding simple tokens. "
    "science space history politics and algorithms shape our world."
)

# Extract unique characters as vocabulary
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Corpus Length: {len(corpus)} characters")
print(f"Vocabulary Size: {vocab_size} unique characters: {''.join(chars)!r}")

# -------------------------------------------------------------
# 2. Build Sliding-Window Dataset
# -------------------------------------------------------------
# Predict the next character given the previous `block_size` characters
block_size = 5  # context window of 5 characters

X, Y = [], []
for i in range(len(corpus) - block_size):
    context = corpus[i : i + block_size]
    target = corpus[i + block_size]
    X.append([char_to_id[ch] for ch in context])
    Y.append(char_to_id[target])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(f"Created {len(X)} training samples. Example: '{corpus[:block_size]}' -> '{corpus[block_size]}'")

# -------------------------------------------------------------
# 3. Model Architecture: Simple Character-Level MLP
# -------------------------------------------------------------
class SimpleCharLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, hidden_dim):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Flatten concatenated embeddings: (block_size * embedding_dim) -> hidden_dim
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        # Hidden layer -> output vocabulary logits
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx):
        # idx shape: (Batch, block_size)
        emb = self.embedding(idx)  # (Batch, block_size, embedding_dim)
        emb_flat = emb.view(emb.size(0), -1)  # (Batch, block_size * embedding_dim)
        h = self.activation(self.fc1(emb_flat))  # (Batch, hidden_dim)
        logits = self.fc2(h)  # (Batch, vocab_size)
        return logits

    def generate(self, prompt, max_new_tokens=60, temperature=0.8):
        self.eval()
        # Ensure prompt has at least block_size characters
        prompt_clean = prompt.lower()
        if len(prompt_clean) < self.block_size:
            prompt_clean = prompt_clean.rjust(self.block_size, ' ')
        
        # Take the last block_size characters
        context = [char_to_id.get(ch, 0) for ch in prompt_clean[-self.block_size:]]
        generated = list(prompt_clean)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([context[-self.block_size:]], dtype=torch.long)
                logits = self(inp)[0]  # shape: (vocab_size)
                # Scale logits by temperature: lower = more confident, higher = more random
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                context.append(next_id)
                generated.append(id_to_char[next_id])
                
        return "".join(generated)

# -------------------------------------------------------------
# 4. Training Loop with MLflow Experiment Tracking
# -------------------------------------------------------------
embedding_dim = 16
hidden_dim = 64
learning_rate = 0.01
epochs = 300

model = SimpleCharLM(vocab_size, embedding_dim, block_size, hidden_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Set MLflow experiment
mlflow.set_experiment("Proton_Phase0_Foundations")

print("\nStarting Training & Logging to MLflow...")
with mlflow.start_run(run_name="Char_MLP_Baseline"):
    # Log hyperparameters
    mlflow.log_params({
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "block_size": block_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "total_parameters": sum(p.numel() for p in model.parameters())
    })

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        # Log metric to MLflow every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            perplexity = torch.exp(loss).item()
            mlflow.log_metrics({"loss": loss.item(), "perplexity": perplexity}, step=epoch)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Perplexity: {torch.exp(loss).item():.2f}")

    # Generate sample completions
    sample_prompt = "the p"
    completion = model.generate(sample_prompt, max_new_tokens=80, temperature=0.7)
    print("\n--- Model Generation Sample ---")
    print(f"Prompt: '{sample_prompt}'")
    print(f"Completion: {completion}")
    print("-------------------------------")

    # Save checkpoint artifact
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase0_char_mlp.pt")
    torch.save(model.state_dict(), checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    print(f"\nModel checkpoint saved and logged to MLflow: {checkpoint_path}")

print("\nTraining complete! To view your dashboard, run: mlflow ui")
