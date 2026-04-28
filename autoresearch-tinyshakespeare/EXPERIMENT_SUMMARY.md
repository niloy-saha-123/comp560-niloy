# Autoresearch Experiment Summary

Repo:
- `autoresearch/comp560_tinyshakespeare`

Setup:
- fixed 120s training budget
- fixed Tiny Shakespeare eval batches
- one mutable file: `train.py`
- metric: validation loss
- follow-up: 5-seed baseline-vs-best confirmation run

Baseline:
- `baseline`
- val loss: `2.0558`

Best run:
- `s160_lr4_wd1e3_e400`
- val loss: `1.8888`
- improvement vs baseline: `0.1670`
- relative improvement: `8.12%`

Best config:
- embedding size: `160`
- heads: `5`
- layers: `5`
- dropout: `0.1`
- learning rate: `0.004`
- weight decay: `0.001`
- eval interval: `400`

Top runs:

| Rank | Run name | Val loss |
|------|----------|----------|
| 1 | `s160_lr4_wd1e3_e400` | `1.8888` |
| 2 | `l5_s160_lr4` | `1.9175` |
| 3 | `l5_wd0001` | `1.9216` |
| 4 | `l5_eval400` | `1.9247` |
| 5 | `s160_lr4_wd1e3` | `1.9316` |

Main takeaways:
- smaller model beat larger model under fixed time budget
- lower dropout helped
- lower weight decay helped
- higher learning rate helped
- evaluating less often helped because more training steps fit into same budget
- bigger width and bigger batch were worse here

Confirmation run:

| Config | Mean val loss | Std val loss | Mean tok/s | Mean steps |
|--------|---------------|--------------|------------|------------|
| `baseline_confirm` | `2.2846` | `0.1735` | `6173.32` | `181.4` |
| `best_confirm` | `2.1181` | `0.0699` | `8559.68` | `251.4` |

Confirmed takeaway:
- best config improved mean val loss by `0.1665`
- relative improvement: `7.29%`
- best config also ran faster and more consistently across seeds
- CPU-only runs showed high variance for baseline, so multi-seed comparison mattered

Budget sweep:

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

Budget sweep takeaway:
- best config won at every tested budget
- `60s`: best beat baseline by `0.1421` (`5.87%`)
- `120s`: best beat baseline by `0.1247` (`5.70%`)
- `240s`: best beat baseline by `0.1390` (`7.26%`)
- bigger model never caught up
- fixed-budget winner stayed same even when budget grew `4x`

Raw run log:
- `results.jsonl`
- `seed_compare_results.jsonl`
- `budget_sweep_results.jsonl`
