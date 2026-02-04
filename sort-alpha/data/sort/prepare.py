"""
Prepare a dataset for the 'Alphabetic Sort' experiment.
Input: Random string of characters (e.g., 'dbca') followed by ':'
Target: Sorted string (e.g., 'abcd') followed by newline.

Example:
dbca:abcd
"""
import os
import pickle
import random
import numpy as np

def get_sorted_pair(length):
    # Generate random string
    chars = [chr(random.randint(ord('a'), ord('z'))) for _ in range(length)]
    input_str = "".join(chars)
    # Sort it
    target_str = "".join(sorted(chars))
    return f"{input_str}:{target_str}\n"

# Configuration
num_samples = 10000
min_len = 3
max_len = 5

# Generate dataset
data_parts = []
for _ in range(num_samples):
    length = random.randint(min_len, max_len)
    data_parts.append(get_sorted_pair(length))

data = "".join(data_parts)

print(f"Generated {len(data):,} characters of data")
print("First 5 samples:")
print("\n".join(data.split('\n')[:5]))

# Creating vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Chars: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Split into train/val
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")

# Save
os.makedirs(os.path.dirname(__file__), exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
