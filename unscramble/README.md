# Word Unscrambling Experiment

Train a tiny GPT to unscramble words character-by-character.

## Task
Map scrambled words to correct spellings.
- 3-letter: `tca: cat`
- 4-letter: `ursn: runs`

## Experiment Log

### Run 1: 3-letter words
**Purpose:** Verify workflow and model capacity on simple 3-letter permutations.

**Config:**
- `dataset`: basic (3-letter words)
- `max_iters`: 2000
- `device`: mps
- `n_layer`: 3, `n_head`: 3, `n_embd`: 120

**Results:**
- **Final Loss**: ~0.60
- **Performance**: The model learned the mapping perfectly for the small vocabulary.

**Sample Output:**
`![Terminal Output](terminal_output_basic.png)`
```
edf: fed
rac: car
nuf: fun
cna: can
atr: tar
```

### Run 2: 4-letter words
**Purpose:** Test model generalization on a slightly larger state space (4-letter permutations).

**Config:**
- `dataset`: advanced (4-letter words)
- `max_iters`: 2000
- `device`: mps
- `n_layer`: 3, `n_head`: 3, `n_embd`: 120

**Results:**
- **Performance**: Excellent. The model correctly unscrambles 4-letter words (verified via sampling).
- **Loss**: Low (implied by perfect sampling accuracy).

**Sample Output:**
`![Terminal Output](terminal_output_advanced.png)`
```
ehom: home
nsca: cans
ansc: cans
arst: star
pkra: park
atsr: rats
jasr: jars
```

## WandB Training Graphs
`![WandB Loss Graph](wandb_loss.png)`

**Observations:**
*   The **val/loss** graph shows that the model learns the 3-letter task (Basic) slightly faster and achieves a lower final loss.
*   This is expected as the 4-letter state space is larger, but both models successfully converge to a low loss (< 0.7).