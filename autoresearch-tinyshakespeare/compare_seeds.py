from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
RESULTS = ROOT / "seed_compare_results.jsonl"


CONFIGS = {
    "baseline_confirm": {
        "AR_N_EMBD": "192",
        "AR_N_HEAD": "6",
        "AR_N_LAYER": "6",
        "AR_DROPOUT": "0.2",
        "AR_LR": "0.003",
        "AR_WEIGHT_DECAY": "0.01",
        "AR_EVAL_INTERVAL": "200",
    },
    "best_confirm": {
        "AR_N_EMBD": "160",
        "AR_N_HEAD": "5",
        "AR_N_LAYER": "5",
        "AR_DROPOUT": "0.1",
        "AR_LR": "0.004",
        "AR_WEIGHT_DECAY": "0.001",
        "AR_EVAL_INTERVAL": "400",
    },
}


def parse_last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("No JSON summary found in train.py output")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run_one(name: str, seed: int, time_budget_s: int) -> dict:
    env = os.environ.copy()
    env.update(CONFIGS[name])
    env["AR_RUN_NAME"] = f"{name}_seed{seed}"
    env["AR_SEED"] = str(seed)
    env["AR_TIME_BUDGET"] = str(time_budget_s)
    proc = subprocess.run(
        [sys.executable, str(TRAIN)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    summary = parse_last_json(proc.stdout)
    summary["comparison_group"] = name
    return summary


def main() -> None:
    seeds = [1337, 2024, 7, 11, 42]
    time_budget_s = 120
    all_rows: list[dict] = []

    for name in ("baseline_confirm", "best_confirm"):
        print(f"\n=== {name} ===", flush=True)
        for seed in seeds:
            print(f"\n--- seed {seed} ---", flush=True)
            row = run_one(name, seed, time_budget_s)
            all_rows.append(row)

    RESULTS.write_text(
        "\n".join(json.dumps(row) for row in all_rows) + "\n",
        encoding="utf-8",
    )

    print("\nSUMMARY", flush=True)
    for name in ("baseline_confirm", "best_confirm"):
        rows = [r for r in all_rows if r["comparison_group"] == name]
        vals = [r["val_loss"] for r in rows]
        tps = [r["tokens_per_second"] for r in rows]
        steps = [r["step"] for r in rows]
        print(
            json.dumps(
                {
                    "name": name,
                    "n": len(rows),
                    "mean_val_loss": round(mean(vals), 6),
                    "std_val_loss": round(std(vals), 6),
                    "mean_tokens_per_second": round(mean(tps), 2),
                    "mean_steps": round(mean(steps), 2),
                    "best_val_loss": round(min(vals), 6),
                    "worst_val_loss": round(max(vals), 6),
                }
            ),
            flush=True,
        )

    base_vals = [r["val_loss"] for r in all_rows if r["comparison_group"] == "baseline_confirm"]
    best_vals = [r["val_loss"] for r in all_rows if r["comparison_group"] == "best_confirm"]
    improvement = mean(base_vals) - mean(best_vals)
    rel = 100.0 * improvement / mean(base_vals)
    print(
        json.dumps(
            {
                "delta_mean_val_loss": round(improvement, 6),
                "relative_improvement_percent": round(rel, 4),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
