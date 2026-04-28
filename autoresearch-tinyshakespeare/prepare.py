from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "input.txt"
META_PATH = ROOT / "meta.json"
BATCH_PATH = ROOT / "eval_batches.npz"

BATCH_SIZE = 32
BLOCK_SIZE = 128
EVAL_ITERS = 40
SEED = 1337


def ensure_data() -> str:
    if not DATA_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def make_batches(data: np.ndarray, batch_size: int, block_size: int, n_batches: int, seed: int):
    rng = np.random.RandomState(seed)
    max_i = len(data) - block_size - 1
    xs = np.zeros((n_batches, batch_size, block_size), dtype=np.int64)
    ys = np.zeros((n_batches, batch_size, block_size), dtype=np.int64)
    for b in range(n_batches):
        ix = rng.randint(0, max_i, size=(batch_size,))
        for j, i in enumerate(ix):
            xs[b, j] = data[i : i + block_size]
            ys[b, j] = data[i + 1 : i + block_size + 1]
    return xs, ys


def main() -> None:
    text = ensure_data()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    val_data = encoded[n:]

    train_x, train_y = make_batches(train_data, BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS, SEED)
    val_x, val_y = make_batches(val_data, BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS, SEED + 1)

    np.savez_compressed(BATCH_PATH, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
    META_PATH.write_text(
        json.dumps(
            {
                "vocab_size": len(chars),
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "eval_iters": EVAL_ITERS,
                "train_tokens": int(len(train_data)),
                "val_tokens": int(len(val_data)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("prepared", DATA_PATH)
    print("meta", META_PATH)
    print("eval_batches", BATCH_PATH)


if __name__ == "__main__":
    main()
