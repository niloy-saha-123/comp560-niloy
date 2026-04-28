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
