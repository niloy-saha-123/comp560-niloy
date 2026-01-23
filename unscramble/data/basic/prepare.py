"""
Prepare word unscrambling dataset for character-level language modeling.
Creates data like: "tca: cat" where scrambled word is mapped to original.
"""
import os
import pickle
import random
import numpy as np

# List of simple 3-letter words to use
words = [
    'cat', 'dog', 'rat', 'bat', 'hat', 'mat', 'sat', 'fat',
    'car', 'bar', 'jar', 'tar', 'war', 'van', 'can', 'man',
    'run', 'sun', 'fun', 'bun', 'gun', 'pen', 'hen', 'ten',
    'red', 'bed', 'fed', 'led', 'wed', 'box', 'fox', 'mix'
]

def scramble(word):
    """Scramble letters of a word. Keep trying until it's different from original."""
    letters = list(word)
    scrambled = word
    attempts = 0
    while scrambled == word and attempts < 10: # shuffle and make sure the scrambled word is different from the original
        random.shuffle(letters)
        scrambled = ''.join(letters)
        attempts += 1
    return scrambled 

# Generate training data
target_length = 100_000  # ~100KB of training data
lines = []
total_length = 0

# Keep adding scrambled word pairs until target length is reached
while total_length < target_length:
    word = random.choice(words)
    scrambled = scramble(word)
    line = f"{scrambled}: {word}"
    lines.append(line)
    total_length += len(line) + 1  # +1 for newline

data = '\n'.join(lines)

print(f"First 20 lines of data:")
for i in range(min(20, len(lines))):
    print(lines[i])

print(f"\nlength of dataset in characters: {len(data):,}")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}")
print(f"vocab size: {vocab_size:,}")

# Create character-to-integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Create train/val split (90/10), 10% for validation
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Encode to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nData preparation complete!")