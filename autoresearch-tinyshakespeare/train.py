from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "meta.json"
BATCH_PATH = ROOT / "eval_batches.npz"
DATA_PATH = ROOT / "input.txt"
RESULTS_PATH = ROOT / "results.jsonl"
run_name = os.environ.get("AR_RUN_NAME", "baseline")


# Only section agent should edit.
batch_size = int(os.environ.get("AR_BATCH_SIZE", 32))
block_size = int(os.environ.get("AR_BLOCK_SIZE", 128))
n_embd = int(os.environ.get("AR_N_EMBD", 192))
n_head = int(os.environ.get("AR_N_HEAD", 6))
n_layer = int(os.environ.get("AR_N_LAYER", 6))
dropout = float(os.environ.get("AR_DROPOUT", 0.2))
learning_rate = float(os.environ.get("AR_LR", 3e-3))
weight_decay = float(os.environ.get("AR_WEIGHT_DECAY", 0.01))
grad_clip = float(os.environ.get("AR_GRAD_CLIP", 1.0))
eval_interval = int(os.environ.get("AR_EVAL_INTERVAL", 200))
time_budget_s = int(os.environ.get("AR_TIME_BUDGET", 180))
seed = int(os.environ.get("AR_SEED", 1337))


device = (
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
torch.manual_seed(seed)


meta = json.loads(META_PATH.read_text(encoding="utf-8"))
fixed = np.load(BATCH_PATH)
text = DATA_PATH.read_text(encoding="utf-8")
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
vocab_size = meta["vocab_size"]


def get_batch():
    ix = torch.randint(len(train_data) - block_size - 1, (batch_size,))
    x = torch.stack([train_data[i : i + block_size] for i in ix])
    y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def fixed_eval(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        xs = fixed[f"{split}_x"]
        ys = fixed[f"{split}_y"]
        losses = torch.zeros(xs.shape[0], device=device)
        for i in range(xs.shape[0]):
            xb = torch.from_numpy(xs[i]).to(device)
            yb = torch.from_numpy(ys[i]).to(device)
            _, loss = model(xb, yb)
            losses[i] = loss
        out[split] = float(losses.mean().item())
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

step = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < time_budget_s:
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    step += 1
    if step == 1 or step % eval_interval == 0:
        metrics = fixed_eval(model)
        elapsed = time.perf_counter() - t0
        print(
            f"step {step:5d} | train {metrics['train']:.4f} | "
            f"val {metrics['val']:.4f} | elapsed {elapsed:.1f}s",
            flush=True,
        )

elapsed = time.perf_counter() - t0
metrics = fixed_eval(model)
summary = {
    "run_name": run_name,
    "seed": seed,
    "step": step,
    "train_loss": metrics["train"],
    "val_loss": metrics["val"],
    "elapsed_s": elapsed,
    "tokens_per_second": (step * batch_size * block_size) / max(elapsed, 1e-9),
    "device": device,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "grad_clip": grad_clip,
    "batch_size": batch_size,
    "block_size": block_size,
    "eval_interval": eval_interval,
    "time_budget_s": time_budget_s,
}
with RESULTS_PATH.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary) + "\n")
print(json.dumps(summary))
