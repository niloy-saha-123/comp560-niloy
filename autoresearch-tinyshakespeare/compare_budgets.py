from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
RESULTS = ROOT / "budget_sweep_results.jsonl"
SUMMARY = ROOT / "BUDGET_SWEEP_SUMMARY.md"


CONFIGS = {
    "baseline": {
        "AR_N_EMBD": "192",
        "AR_N_HEAD": "6",
        "AR_N_LAYER": "6",
        "AR_DROPOUT": "0.2",
        "AR_LR": "0.003",
        "AR_WEIGHT_DECAY": "0.01",
        "AR_EVAL_INTERVAL": "200",
    },
    "best": {
        "AR_N_EMBD": "160",
        "AR_N_HEAD": "5",
        "AR_N_LAYER": "5",
        "AR_DROPOUT": "0.1",
        "AR_LR": "0.004",
        "AR_WEIGHT_DECAY": "0.001",
        "AR_EVAL_INTERVAL": "400",
    },
    "bigger": {
        "AR_N_EMBD": "224",
        "AR_N_HEAD": "7",
        "AR_N_LAYER": "6",
        "AR_DROPOUT": "0.1",
        "AR_LR": "0.003",
        "AR_WEIGHT_DECAY": "0.01",
        "AR_EVAL_INTERVAL": "200",
    },
}


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_name_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


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


def run_one(config_name: str, budget_s: int, seed: int) -> dict:
    env = os.environ.copy()
    env.update(CONFIGS[config_name])
    env["AR_RUN_NAME"] = f"{config_name}_budget{budget_s}_seed{seed}"
    env["AR_SEED"] = str(seed)
    env["AR_TIME_BUDGET"] = str(budget_s)
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
    summary["budget_s"] = budget_s
    summary["comparison_group"] = config_name
    return summary


def build_summary(rows: list[dict], budgets: list[int], config_names: list[str]) -> str:
    lines = [
        "# Budget Sweep Summary",
        "",
        "Purpose:",
        "- compare baseline vs best vs bigger config across multiple fixed budgets",
        "- check whether best short-budget config stays best as budget grows",
        "",
        "Configs:",
        "- `baseline`: 192 embd / 6 heads / 6 layers / dropout 0.2 / lr 0.003 / wd 0.01",
        "- `best`: 160 embd / 5 heads / 5 layers / dropout 0.1 / lr 0.004 / wd 0.001",
        "- `bigger`: 224 embd / 7 heads / 6 layers / dropout 0.1 / lr 0.003 / wd 0.01",
        "",
        "| Config | Budget (s) | Mean val loss | Std val loss | Mean tok/s | Mean steps |",
        "|--------|------------|---------------|--------------|------------|------------|",
    ]
    for budget_s in budgets:
        subset_budget = [r for r in rows if r["budget_s"] == budget_s]
        for config_name in config_names:
            subset = [r for r in subset_budget if r["comparison_group"] == config_name]
            vals = [r["val_loss"] for r in subset]
            tps = [r["tokens_per_second"] for r in subset]
            steps = [r["step"] for r in subset]
            lines.append(
                f"| `{config_name}` | `{budget_s}` | "
                f"`{mean(vals):.4f}` | `{std(vals):.4f}` | "
                f"`{mean(tps):.2f}` | `{mean(steps):.1f}` |"
            )
        best_row = min(subset_budget, key=lambda r: r["val_loss"])
        lines.extend(
            [
                "",
                f"Best single run at `{budget_s}s`: `{best_row['run_name']}` with val loss `{best_row['val_loss']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    budgets = parse_int_list("AR_COMPARE_BUDGETS", [60, 120, 240])
    seeds = parse_int_list("AR_COMPARE_SEEDS", [1337, 2024, 7])
    config_names = parse_name_list("AR_COMPARE_CONFIGS", list(CONFIGS.keys()))
    rows: list[dict] = []

    for budget_s in budgets:
        print(f"\n=== budget {budget_s}s ===", flush=True)
        for config_name in config_names:
            print(f"\n--- config {config_name} ---", flush=True)
            for seed in seeds:
                print(f"\nseed {seed}", flush=True)
                row = run_one(config_name, budget_s, seed)
                rows.append(row)

    RESULTS.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    SUMMARY.write_text(build_summary(rows, budgets, config_names), encoding="utf-8")

    print("\nSUMMARY", flush=True)
    for budget_s in budgets:
        print(json.dumps({"budget_s": budget_s}), flush=True)
        subset_budget = [r for r in rows if r["budget_s"] == budget_s]
        for config_name in config_names:
            subset = [r for r in subset_budget if r["comparison_group"] == config_name]
            vals = [r["val_loss"] for r in subset]
            tps = [r["tokens_per_second"] for r in subset]
            steps = [r["step"] for r in subset]
            print(
                json.dumps(
                    {
                        "config": config_name,
                        "mean_val_loss": round(mean(vals), 6),
                        "std_val_loss": round(std(vals), 6),
                        "mean_tokens_per_second": round(mean(tps), 2),
                        "mean_steps": round(mean(steps), 2),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
