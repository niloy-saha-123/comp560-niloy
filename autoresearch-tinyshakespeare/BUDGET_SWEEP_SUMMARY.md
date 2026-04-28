# Budget Sweep Summary

Purpose:
- compare baseline vs best vs bigger config across multiple fixed budgets
- check whether best short-budget config stays best as budget grows

Configs:
- `baseline`: 192 embd / 6 heads / 6 layers / dropout 0.2 / lr 0.003 / wd 0.01
- `best`: 160 embd / 5 heads / 5 layers / dropout 0.1 / lr 0.004 / wd 0.001
- `bigger`: 224 embd / 7 heads / 6 layers / dropout 0.1 / lr 0.003 / wd 0.01

| Config | Budget (s) | Mean val loss | Std val loss | Mean tok/s | Mean steps |
|--------|------------|---------------|--------------|------------|------------|
| `baseline` | `60` | `2.4208` | `0.0075` | `8581.65` | `126.3` |
| `best` | `60` | `2.2787` | `0.1162` | `11240.36` | `165.3` |
| `bigger` | `60` | `2.5252` | `0.0780` | `4587.53` | `67.7` |

Best single run at `60s`: `best_budget60_seed7` with val loss `2.1866`

| `baseline` | `120` | `2.1874` | `0.0537` | `8004.37` | `235.0` |
| `best` | `120` | `2.0627` | `0.0510` | `9931.72` | `291.7` |
| `bigger` | `120` | `2.3497` | `0.1145` | `5135.77` | `151.3` |

Best single run at `120s`: `best_budget120_seed2024` with val loss `2.0064`

| `baseline` | `240` | `1.9163` | `0.0190` | `8555.54` | `501.7` |
| `best` | `240` | `1.7772` | `0.0112` | `12913.85` | `757.7` |
| `bigger` | `240` | `1.9548` | `0.0559` | `6498.78` | `381.0` |

Best single run at `240s`: `best_budget240_seed2024` with val loss `1.7700`

