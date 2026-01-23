# Word Unscrambling Experiment

Train a tiny GPT to unscramble words character-by-character.

## Task

Learn to map scrambled words to correct spellings:
- 3-letter: `tca: cat`, `ogd: dog`, `nru: run`
- 4-letter: `ursn: runs`, `satr: rats`, `bsra: bars`

## Experiments

### Basic (3-letter words)
- **Vocabulary:** 32 common 3-letter words
- **Training:** 2000 iterations (on M4)
- **Results:** ~100% correct unscrambles
- **Model:** 3-layer GPT, 520K parameters

### Advanced (4-letter words)
- **Vocabulary:** 21 common 4-letter words
- **Training:** 2000 iterations (on M4)
- **Results:** ~100% correct unscrambles

## Key Finding

**Both 4-letter and 3-letter words gave correct results**, will try with more complex and bigger dataset in future

## Run

```bash
# Basic (3-letter)
python data/basic/prepare.py
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --device=cpu --num_samples=3 --max_new_tokens=100

# Advanced (4-letter)
python data/advanced/prepare.py
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/advanced.py
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/advanced.py --device=cpu --num_samples=3 --max_new_tokens=150