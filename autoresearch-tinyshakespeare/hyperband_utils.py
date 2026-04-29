from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"

HEAD_MAP = {
    128: 4,
    160: 5,
    192: 6,
    224: 7,
}

SEARCH_SPACE = {
    "n_embd": [128, 160, 192, 224],
    "n_layer": [4, 5, 6],
    "dropout": [0.05, 0.10, 0.15, 0.20],
    "learning_rate": [0.002, 0.003, 0.004, 0.005],
    "weight_decay": [0.0, 0.001, 0.01],
    "eval_interval": [200, 400],
}

FIXED_DEFAULTS = {
    "batch_size": 32,
    "block_size": 128,
    "grad_clip": 1.0,
}

BASELINE_CONFIG = {
    "n_embd": 192,
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.2,
    "learning_rate": 0.003,
    "weight_decay": 0.01,
    "eval_interval": 200,
    **FIXED_DEFAULTS,
}

MANUAL_BEST_CONFIG = {
    "n_embd": 160,
    "n_head": 5,
    "n_layer": 5,
    "dropout": 0.1,
    "learning_rate": 0.004,
    "weight_decay": 0.001,
    "eval_interval": 400,
    **FIXED_DEFAULTS,
}

BIGGER_CONFIG = {
    "n_embd": 224,
    "n_head": 7,
    "n_layer": 6,
    "dropout": 0.1,
    "learning_rate": 0.003,
    "weight_decay": 0.01,
    "eval_interval": 200,
    **FIXED_DEFAULTS,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def canonical_config(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha1(canonical_config(config).encode("utf-8")).hexdigest()
    return digest[:10]


def derive_n_head(n_embd: int) -> int:
    if n_embd not in HEAD_MAP:
        raise ValueError(f"Unsupported n_embd={n_embd}")
    return HEAD_MAP[n_embd]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg["n_head"] = derive_n_head(int(cfg["n_embd"]))
    for key, value in FIXED_DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg


def sample_config(rng: random.Random) -> dict[str, Any]:
    config = {
        "n_embd": rng.choice(SEARCH_SPACE["n_embd"]),
        "n_layer": rng.choice(SEARCH_SPACE["n_layer"]),
        "dropout": rng.choice(SEARCH_SPACE["dropout"]),
        "learning_rate": rng.choice(SEARCH_SPACE["learning_rate"]),
        "weight_decay": rng.choice(SEARCH_SPACE["weight_decay"]),
        "eval_interval": rng.choice(SEARCH_SPACE["eval_interval"]),
    }
    return validate_config(config)


def sample_unique_configs(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    while len(out) < n:
        cfg = sample_config(rng)
        key = canonical_config(cfg)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def config_to_env(config: dict[str, Any]) -> dict[str, str]:
    cfg = validate_config(config)
    return {
        "AR_BATCH_SIZE": str(cfg["batch_size"]),
        "AR_BLOCK_SIZE": str(cfg["block_size"]),
        "AR_N_EMBD": str(cfg["n_embd"]),
        "AR_N_HEAD": str(cfg["n_head"]),
        "AR_N_LAYER": str(cfg["n_layer"]),
        "AR_DROPOUT": str(cfg["dropout"]),
        "AR_LR": str(cfg["learning_rate"]),
        "AR_WEIGHT_DECAY": str(cfg["weight_decay"]),
        "AR_GRAD_CLIP": str(cfg["grad_clip"]),
        "AR_EVAL_INTERVAL": str(cfg["eval_interval"]),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def bracket_schedule(s: int, eta: int = 3, r: int = 30, R: int = 270) -> list[dict[str, int]]:
    s_max = int(math.log(R / r, eta))
    B = (s_max + 1) * R
    n = math.ceil((B / R) * (eta**s) / (s + 1))
    r0 = int(R * (eta ** (-s)))
    stages = []
    for i in range(s + 1):
        n_i = math.floor(n * (eta ** (-i)))
        r_i = int(r0 * (eta**i))
        stages.append({"stage": i, "n": n_i, "budget_s": r_i})
    return stages


def run_train_once(
    config: dict[str, Any],
    budget_s: int,
    seed: int,
    run_name: str,
    result_json_path: Path,
    results_path: Path,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(config_to_env(config))
    env["AR_RUN_NAME"] = run_name
    env["AR_SEED"] = str(seed)
    env["AR_TIME_BUDGET"] = str(budget_s)
    env["AR_RESULT_JSON_PATH"] = str(result_json_path)
    env["AR_RESULTS_PATH"] = str(results_path)
    if checkpoint_path is not None:
        env["AR_CKPT_PATH"] = str(checkpoint_path)
    if resume_from is not None:
        env["AR_RESUME_FROM"] = str(resume_from)

    ensure_parent(result_json_path)
    ensure_parent(results_path)
    if checkpoint_path is not None:
        ensure_parent(checkpoint_path)

    run_kwargs = {
        "cwd": ROOT,
        "env": env,
        "check": True,
        "text": True,
    }
    if verbose:
        proc = subprocess.run([sys.executable, str(TRAIN)], **run_kwargs)
    else:
        proc = subprocess.run(
            [sys.executable, str(TRAIN)],
            capture_output=True,
            **run_kwargs,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
    return json.loads(result_json_path.read_text(encoding="utf-8"))


def rank_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["val_loss"], -row["tokens_per_second"], -row["step"]))


def run_successive_halving_bracket(
    s: int,
    artifacts_root: Path,
    bracket_seed: int,
    train_seed: int = 1337,
    eta: int = 3,
    r: int = 30,
    R: int = 270,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts_root = ensure_dir(artifacts_root)
    stage_results_dir = ensure_dir(artifacts_root / "stage_results")
    ckpt_dir = ensure_dir(artifacts_root / "checkpoints" / f"s{s}")
    raw_results_dir = ensure_dir(artifacts_root / "raw_train_logs")
    hyperband_results_path = artifacts_root / "hyperband_results.jsonl"

    schedule = bracket_schedule(s=s, eta=eta, r=r, R=R)
    initial_configs = sample_unique_configs(schedule[0]["n"], seed=bracket_seed)
    candidates = [
        {
            "config": cfg,
            "config_id": config_id(cfg),
            "seed": train_seed,
            "checkpoint_path": ckpt_dir / f"{config_id(cfg)}.pt",
        }
        for cfg in initial_configs
    ]

    all_rows: list[dict[str, Any]] = []
    bracket_log = {
        "bracket_s": s,
        "eta": eta,
        "r": r,
        "R": R,
        "train_seed": train_seed,
        "bracket_seed": bracket_seed,
        "schedule": schedule,
        "stages": [],
    }

    for stage_idx, stage in enumerate(schedule):
        previous_budget_s = schedule[stage_idx - 1]["budget_s"] if stage_idx > 0 else 0
        marginal_budget_s = stage["budget_s"] - previous_budget_s
        stage_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            run_name = f"hb_s{s}_i{stage_idx}_{candidate['config_id']}"
            result_json_path = stage_results_dir / f"{run_name}.json"
            raw_results_path = raw_results_dir / f"{run_name}.jsonl"
            row = run_train_once(
                config=candidate["config"],
                budget_s=stage["budget_s"],
                seed=candidate["seed"],
                run_name=run_name,
                result_json_path=result_json_path,
                results_path=raw_results_path,
                checkpoint_path=candidate["checkpoint_path"],
                resume_from=candidate["checkpoint_path"] if stage_idx > 0 else None,
                verbose=verbose,
            )
            row["config"] = validate_config(candidate["config"])
            row["config_id"] = candidate["config_id"]
            row["comparison_group"] = "hyperband"
            row["bracket_s"] = s
            row["stage"] = stage_idx
            row["stage_budget_s"] = stage["budget_s"]
            row["marginal_budget_s"] = marginal_budget_s
            row["bracket_seed"] = bracket_seed
            row["train_seed"] = train_seed
            row["checkpoint_path"] = str(candidate["checkpoint_path"])
            append_jsonl(hyperband_results_path, row)
            stage_rows.append(row)
            all_rows.append(row)

        ranked = rank_results(stage_rows)
        promote_n = schedule[stage_idx + 1]["n"] if stage_idx + 1 < len(schedule) else 0
        promoted_ids = [row["config_id"] for row in ranked[:promote_n]]
        bracket_log["stages"].append(
            {
                "stage": stage_idx,
                "budget_s": stage["budget_s"],
                "marginal_budget_s": marginal_budget_s,
                "n": len(stage_rows),
                "winner": ranked[0]["run_name"],
                "winner_val_loss": ranked[0]["val_loss"],
                "promoted_ids": promoted_ids,
                "rows": [
                    {
                        "run_name": row["run_name"],
                        "config_id": row["config_id"],
                        "val_loss": row["val_loss"],
                        "tokens_per_second": row["tokens_per_second"],
                        "step": row["step"],
                    }
                    for row in ranked
                ],
            }
        )
        promoted_set = set(promoted_ids)
        candidates = [candidate for candidate in candidates if candidate["config_id"] in promoted_set]

    return all_rows, bracket_log


def write_bracket_log(bracket_logs: list[dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(bracket_logs, indent=2) + "\n", encoding="utf-8")


def compute_best_so_far(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = list(rows)
    cumulative = 0
    best_val = float("inf")
    out = []
    for idx, row in enumerate(ordered, start=1):
        cumulative += row.get("marginal_budget_s", row["time_budget_s"])
        best_val = min(best_val, row["val_loss"])
        out.append(
            {
                "run_index": idx,
                "run_name": row["run_name"],
                "bracket_s": row.get("bracket_s"),
                "stage": row.get("stage"),
                "cumulative_compute_s": cumulative,
                "val_loss": row["val_loss"],
                "best_val_loss_so_far": best_val,
            }
        )
    return out


def bracket_winner_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners = {}
    for row in rows:
        key = (row["bracket_s"], row["stage"])
        if key not in winners or row["val_loss"] < winners[key]["val_loss"]:
            winners[key] = row
    return [
        {
            "bracket_s": key[0],
            "stage": key[1],
            "run_name": row["run_name"],
            "config_id": row["config_id"],
            "val_loss": row["val_loss"],
        }
        for key, row in sorted(winners.items())
    ]


def select_hyperband_best_config(rows: list[dict[str, Any]], require_max_budget: bool = True) -> dict[str, Any]:
    if require_max_budget:
        max_budget = max(row["stage_budget_s"] for row in rows)
        rows = [row for row in rows if row["stage_budget_s"] == max_budget]
    best_row = rank_results(rows)[0]
    return dict(best_row["config"])


def compare_reference_configs(
    config_map: dict[str, dict[str, Any]],
    budgets: list[int],
    seeds: list[int],
    artifacts_root: Path,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    artifacts_root = ensure_dir(artifacts_root)
    compare_results_path = artifacts_root / "hyperband_compare_results.jsonl"
    raw_results_dir = ensure_dir(artifacts_root / "compare_raw")
    result_json_dir = ensure_dir(artifacts_root / "compare_json")
    rows: list[dict[str, Any]] = []

    for budget_s in budgets:
        for name, config in config_map.items():
            for seed in seeds:
                run_name = f"compare_{name}_budget{budget_s}_seed{seed}"
                row = run_train_once(
                    config=config,
                    budget_s=budget_s,
                    seed=seed,
                    run_name=run_name,
                    result_json_path=result_json_dir / f"{run_name}.json",
                    results_path=raw_results_dir / f"{run_name}.jsonl",
                    checkpoint_path=None,
                    resume_from=None,
                    verbose=verbose,
                )
                row["comparison_group"] = name
                row["budget_s"] = budget_s
                row["config"] = validate_config(config)
                append_jsonl(compare_results_path, row)
                rows.append(row)
    return rows


def summarize_by_budget(rows: list[dict[str, Any]], config_order: list[str]) -> list[dict[str, Any]]:
    budgets = sorted({row["budget_s"] for row in rows})
    summary_rows: list[dict[str, Any]] = []
    for budget_s in budgets:
        for config_name in config_order:
            subset = [row for row in rows if row["budget_s"] == budget_s and row["comparison_group"] == config_name]
            summary_rows.append(
                {
                    "budget_s": budget_s,
                    "config": config_name,
                    "mean_val_loss": mean([row["val_loss"] for row in subset]),
                    "std_val_loss": std([row["val_loss"] for row in subset]),
                    "mean_tokens_per_second": mean([row["tokens_per_second"] for row in subset]),
                    "mean_steps": mean([row["step"] for row in subset]),
                }
            )
    return summary_rows


def write_hyperband_summary(
    search_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    hyperband_best_config: dict[str, Any],
    output_path: Path,
) -> None:
    best_so_far = compute_best_so_far(search_rows)
    final_search_best = min(search_rows, key=lambda row: row["val_loss"])
    config_order = sorted({row["comparison_group"] for row in compare_rows})
    budget_summary = summarize_by_budget(compare_rows, config_order=config_order)

    lines = [
        "# Hyperband Summary",
        "",
        "Question:",
        "- does Hyperband beat manual autoresearch sweep under same fixed compute budget?",
        "",
        "Hyperband winner:",
        f"- run: `{final_search_best['run_name']}`",
        f"- val loss: `{final_search_best['val_loss']:.4f}`",
        f"- config: `{canonical_config(hyperband_best_config)}`",
        "",
        "Search progress:",
        f"- runs: `{len(search_rows)}`",
        f"- total staged compute: `{sum(row.get('marginal_budget_s', 0) for row in search_rows)}s`",
        f"- best-so-far final val loss: `{best_so_far[-1]['best_val_loss_so_far']:.4f}`",
        "",
        "| Config | Budget (s) | Mean val loss | Std val loss | Mean tok/s | Mean steps |",
        "|--------|------------|---------------|--------------|------------|------------|",
    ]
    for row in budget_summary:
        lines.append(
            f"| `{row['config']}` | `{row['budget_s']}` | "
            f"`{row['mean_val_loss']:.4f}` | `{row['std_val_loss']:.4f}` | "
            f"`{row['mean_tokens_per_second']:.2f}` | `{row['mean_steps']:.1f}` |"
        )
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_best_so_far(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    series = compute_best_so_far(rows)
    xs = [row["cumulative_compute_s"] for row in series]
    ys = [row["best_val_loss_so_far"] for row in series]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Cumulative search compute (s)")
    plt.ylabel("Best val loss so far")
    plt.title("Hyperband search progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_budget_comparison(summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    budgets = sorted({row["budget_s"] for row in summary_rows})
    configs = sorted({row["config"] for row in summary_rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for config_name in configs:
        subset = [row for row in summary_rows if row["config"] == config_name]
        subset = sorted(subset, key=lambda row: row["budget_s"])
        axes[0].plot(
            [row["budget_s"] for row in subset],
            [row["mean_val_loss"] for row in subset],
            marker="o",
            label=config_name,
        )
        axes[1].plot(
            [row["budget_s"] for row in subset],
            [row["mean_steps"] for row in subset],
            marker="o",
            label=config_name,
        )

    axes[0].set_title("Mean val loss vs budget")
    axes[0].set_xlabel("Budget (s)")
    axes[0].set_ylabel("Mean val loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Mean steps vs budget")
    axes[1].set_xlabel("Budget (s)")
    axes[1].set_ylabel("Mean steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
