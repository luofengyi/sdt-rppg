import argparse
import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_cli_args(arg_map):
    args = []
    for k, v in arg_map.items():
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)
            continue
        args.extend([key, str(v)])
    return args


def build_trials(space_cfg, mode, max_trials, seed):
    random.seed(seed)
    keys = list(space_cfg.keys())
    values = []
    for k in keys:
        v = space_cfg[k]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"param_space['{k}'] must be a non-empty list")
        values.append(v)

    all_combinations = [dict(zip(keys, x)) for x in itertools.product(*values)]
    if mode == "grid":
        if max_trials > 0:
            return all_combinations[:max_trials]
        return all_combinations

    if max_trials <= 0:
        raise ValueError("random mode needs --max-trials > 0")
    if max_trials >= len(all_combinations):
        random.shuffle(all_combinations)
        return all_combinations
    return random.sample(all_combinations, max_trials)


def parse_metrics(output_text):
    fscore_matches = re.findall(r"F-Score:\s*([0-9]+(?:\.[0-9]+)?)", output_text)
    fscore_idx_matches = re.findall(r"F-Score-index:\s*([0-9]+)", output_text)
    best_fscore = float(fscore_matches[-1]) if fscore_matches else float("nan")
    best_epoch = int(fscore_idx_matches[-1]) if fscore_idx_matches else -1
    return best_fscore, best_epoch


def is_close(a, b, tol):
    return abs(float(a) - float(b)) <= float(tol)


def apply_hard_constraints(trials, hard_constraints):
    if not hard_constraints:
        return trials, 0

    enabled = hard_constraints.get("enabled", False)
    if not enabled:
        return trials, 0

    alpha_keys = hard_constraints.get(
        "alpha_keys",
        ["ulgm-alpha-t", "ulgm-alpha-a", "ulgm-alpha-v", "ulgm-alpha-r"],
    )
    require_sum = hard_constraints.get("require_alpha_sum", 1.0)
    tol = hard_constraints.get("alpha_sum_tolerance", 1e-6)
    enforce_non_negative = hard_constraints.get("enforce_alpha_non_negative", True)
    use_rppg_key = hard_constraints.get("use_rppg_key", "use-rppg")
    alpha_r_key = hard_constraints.get("alpha_r_key", "ulgm-alpha-r")
    only_when_normalize = hard_constraints.get("only_when_normalize_alpha", False)
    normalize_key = hard_constraints.get("normalize_alpha_key", "ulgm-normalize-alpha")

    filtered = []
    removed = 0
    for trial in trials:
        use_rppg = bool(trial.get(use_rppg_key, False))
        normalize_alpha = bool(trial.get(normalize_key, False))
        if only_when_normalize and (not normalize_alpha):
            filtered.append(trial)
            continue

        active_alpha_keys = []
        for k in alpha_keys:
            if k == alpha_r_key and (not use_rppg):
                continue
            if k in trial:
                active_alpha_keys.append(k)

        if len(active_alpha_keys) == 0:
            filtered.append(trial)
            continue

        alpha_vals = [float(trial[k]) for k in active_alpha_keys]
        if enforce_non_negative and any(v < 0 for v in alpha_vals):
            removed += 1
            continue

        alpha_sum = sum(alpha_vals)
        if not is_close(alpha_sum, require_sum, tol):
            removed += 1
            continue

        filtered.append(trial)

    return filtered, removed


def main():
    parser = argparse.ArgumentParser(description="One-click hyperparameter tuning for SDT.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tune_iemocap_example.json",
        help="Path to tuning config json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Search mode: grid or random",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Max trials. 0 means full grid in grid mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random mode.")
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run train.py",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train.py",
        help="Training entry script path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fixed_args = cfg.get("fixed_args", {})
    param_space = cfg.get("param_space", {})
    hard_constraints = cfg.get("hard_constraints", {})
    output_dir = cfg.get("output_dir", "tuning_runs")
    run_name = cfg.get("run_name", "iemocap_tune")

    ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{run_name}_{stamp}")
    ensure_dir(run_dir)

    trials = build_trials(param_space, args.mode, args.max_trials, args.seed)
    trials, removed_by_constraints = apply_hard_constraints(trials, hard_constraints)
    if len(trials) == 0:
        raise ValueError("No trial left after hard constraints; check param_space/hard_constraints.")

    csv_path = os.path.join(run_dir, "results.csv")
    json_path = os.path.join(run_dir, "summary.json")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(log_dir)

    fields = ["trial_id", "status", "best_fscore", "best_epoch", "elapsed_sec", "command"] + list(param_space.keys())
    results = []
    best_item = None

    print(f"[TUNE] total trials: {len(trials)}")
    if removed_by_constraints > 0:
        print(f"[TUNE] removed by hard constraints: {removed_by_constraints}")
    print(f"[TUNE] run dir: {run_dir}")

    for i, param_set in enumerate(trials, start=1):
        trial_args = dict(fixed_args)
        trial_args.update(param_set)
        cmd = [args.python_exec, args.train_script] + to_cli_args(trial_args)
        cmd_str = " ".join(cmd)

        print(f"\n[TUNE] trial {i}/{len(trials)}")
        print(f"[TUNE] cmd: {cmd_str}")
        start = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = round(time.time() - start, 2)

        output_text = proc.stdout or ""
        log_path = os.path.join(log_dir, f"trial_{i:03d}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        if proc.returncode == 0:
            best_fscore, best_epoch = parse_metrics(output_text)
            status = "ok"
        else:
            best_fscore, best_epoch = float("nan"), -1
            status = f"failed({proc.returncode})"

        row = {
            "trial_id": i,
            "status": status,
            "best_fscore": best_fscore,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
            "command": cmd_str,
        }
        row.update(param_set)
        results.append(row)

        if status == "ok" and (best_item is None or best_fscore > best_item["best_fscore"]):
            best_item = row
            print(f"[TUNE] new best fscore: {best_fscore:.4f} (trial {i})")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    summary = {
        "config": args.config,
        "mode": args.mode,
        "max_trials": args.max_trials,
        "seed": args.seed,
        "run_dir": run_dir,
        "total_trials": len(trials),
        "removed_by_hard_constraints": removed_by_constraints,
        "best": best_item,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[TUNE] done.")
    print(f"[TUNE] results csv: {csv_path}")
    print(f"[TUNE] summary json: {json_path}")
    if best_item is None:
        print("[TUNE] no successful trial.")
    else:
        print(f"[TUNE] best trial: {best_item['trial_id']}, best fscore: {best_item['best_fscore']:.4f}")


if __name__ == "__main__":
    main()
