# comp560 Tiny Shakespeare

Small autoresearch variant for class project.

Why:
- official repo targets H100 + ClimbMix
- this variant targets Tiny Shakespeare
- same pattern: fixed prep, one mutable train file, short budget

Files:
- `prepare.py` fixed data + eval batches
- `train.py` only file agent should edit
- `program.md` agent instructions
- `hyperband_utils.py` Hyperband search helpers
- `hyperband_search.ipynb` notebook workflow for Colab / VSCode GPU

Run:
```bash
cd comp560_tinyshakespeare
uv run prepare.py
uv run train.py
```

Confirm best vs baseline:
```bash
cd comp560_tinyshakespeare
uv run compare_seeds.py
```

Sweep budgets:
```bash
cd comp560_tinyshakespeare
uv run compare_budgets.py
```

Run Hyperband notebook:
```bash
cd comp560_tinyshakespeare
jupyter notebook hyperband_search.ipynb
```
