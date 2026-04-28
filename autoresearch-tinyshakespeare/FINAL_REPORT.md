# Final Report: Tiny Shakespeare Autoresearch

## Question

Can a small autoresearch-style loop find a better Tiny Shakespeare training setup under a fixed short compute budget?

## Setup

- repo base: `autoresearch`
- class adaptation: `comp560_tinyshakespeare`
- fixed dataset: Tiny Shakespeare
- fixed evaluation batches from `prepare.py`
- one mutable training file: `train.py`
- fixed training budget per run: `120s`
- metric: validation loss

This follows Karpathy's basic autoresearch pattern:

1. keep evaluation fixed
2. change only training code or hyperparameters
3. run short experiment
4. keep change only if metric improves

## Search Phase

I ran 22 short experiments and logged every run in `results.jsonl`.

Starting baseline:

- run: `baseline`
- val loss: `2.0558`

Best run found during search:

- run: `s160_lr4_wd1e3_e400`
- val loss: `1.8888`

Best config:

- `n_embd=160`
- `n_head=5`
- `n_layer=5`
- `dropout=0.1`
- `learning_rate=0.004`
- `weight_decay=0.001`
- `eval_interval=400`

Improvement from baseline to best single run:

- absolute: `0.1670`
- relative: `8.12%`

## Why Follow-up Was Needed

Single-run search results can be noisy, especially on CPU under short time budgets.

So I ran a confirmation study with 5 seeds for:

- original baseline config
- best config found by search

Those results are logged in `seed_compare_results.jsonl`.

## Confirmation Results

| Config | Mean val loss | Std val loss | Mean tok/s | Mean steps |
|--------|---------------|--------------|------------|------------|
| `baseline_confirm` | `2.2846` | `0.1735` | `6173.32` | `181.4` |
| `best_confirm` | `2.1181` | `0.0699` | `8559.68` | `251.4` |

Mean improvement:

- absolute: `0.1665`
- relative: `7.29%`

## Budget Sweep

To test whether this result was only true at `120s`, I ran one more sweep across:

- budgets: `60s`, `120s`, `240s`
- configs: `baseline`, `best`, `bigger`
- seeds: `3`

Those results are logged in `budget_sweep_results.jsonl` and summarized in `BUDGET_SWEEP_SUMMARY.md`.

| Config | Budget (s) | Mean val loss | Std val loss | Mean tok/s | Mean steps |
|--------|------------|---------------|--------------|------------|------------|
| `baseline` | `60` | `2.4208` | `0.0075` | `8581.65` | `126.3` |
| `best` | `60` | `2.2787` | `0.1162` | `11240.36` | `165.3` |
| `bigger` | `60` | `2.5252` | `0.0780` | `4587.53` | `67.7` |
| `baseline` | `120` | `2.1874` | `0.0537` | `8004.37` | `235.0` |
| `best` | `120` | `2.0627` | `0.0510` | `9931.72` | `291.7` |
| `bigger` | `120` | `2.3497` | `0.1145` | `5135.77` | `151.3` |
| `baseline` | `240` | `1.9163` | `0.0190` | `8555.54` | `501.7` |
| `best` | `240` | `1.7772` | `0.0112` | `12913.85` | `757.7` |
| `bigger` | `240` | `1.9548` | `0.0559` | `6498.78` | `381.0` |

Budget-sweep improvement of `best` over `baseline`:

- `60s`: `0.1421` better (`5.87%`)
- `120s`: `0.1247` better (`5.70%`)
- `240s`: `0.1390` better (`7.26%`)

This was important because it tested whether the "small fast model wins" story was only true at one short budget. It was not. The same config stayed best even when the training budget was multiplied by four.

## Main Findings

1. Autoresearch loop worked on class-scale setup.
   It found a clearly better configuration than starting baseline.

2. Bigger model was not best under fixed compute budget.
   Smaller 5-layer, 160-width model performed better because it trained faster and took more optimizer steps in same wall-clock time.

3. Lower regularization helped.
   Reducing dropout from `0.2` to `0.1` and reducing weight decay from `0.01` to `0.001` both helped.

4. Slightly higher learning rate helped.
   `0.004` worked better than baseline `0.003` in best confirmed config.

5. Less frequent evaluation helped.
   `eval_interval=400` reduced evaluation overhead and allowed more training steps inside same 120-second budget.

6. Multi-seed confirmation mattered.
   Baseline showed much higher variability across seeds than best config, so mean/std comparison gave a more defensible conclusion than one single lucky run.

7. Budget sweep strengthened conclusion.
   Best config won at `60s`, `120s`, and `240s`, while bigger model never caught up.

## Final Conclusion

For this Tiny Shakespeare autoresearch setup, best strategy under a fixed short compute budget was not to make model larger.  
Best strategy was to use a slightly smaller, faster model with lower dropout, lower weight decay, a slightly higher learning rate, and less frequent evaluation.

That changed validation loss from:

- `2.0558` in original baseline search run
- to `1.8888` in best single run

and from:

- `2.2846` mean over 5 baseline seeds
- to `2.1181` mean over 5 best-config seeds

This is enough to support concrete final claim:

**Under fixed compute budget, simpler faster model can outperform larger baseline, and autoresearch loop can discover that automatically.**

## Limitations

- runs were CPU-only
- MPS backend was built in PyTorch but unavailable at runtime
- budget was very short (`120s`), so conclusion is about short-budget optimization, not full-scale final model quality

## Files

- setup instructions: `README.md`
- frozen evaluation: `prepare.py`
- mutable training loop: `train.py`
- search log: `results.jsonl`
- confirmation log: `seed_compare_results.jsonl`
- budget sweep log: `budget_sweep_results.jsonl`
- budget sweep summary: `BUDGET_SWEEP_SUMMARY.md`
- short summary: `EXPERIMENT_SUMMARY.md`
