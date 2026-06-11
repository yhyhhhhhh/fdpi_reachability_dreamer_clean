from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from tools import eval_comprehensive as base_eval


simulation_app = None


def _launch_isaac(headless=True):
    global simulation_app
    base_eval._launch_isaac(headless=headless)
    simulation_app = base_eval.simulation_app


def _checkpoint_step(path):
    match = re.search(r"full_state(?:_v5|_v4)?_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _infer_latest_full_checkpoint(checkpoint_dir):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    candidates = []
    for name in os.listdir(checkpoint_dir):
        match = re.match(r"full_state(?:_v5|_v4)?_(\d+)\.pth$", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))
    if not candidates:
        raise FileNotFoundError(f"No full_state*_*.pth files found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])


def _episode_age(rollout):
    total, num_envs = rollout.done.shape
    age = np.zeros((total, num_envs), dtype=np.int32)
    for env_idx in range(num_envs):
        current = 0
        for t in range(total):
            if rollout.is_first[t, env_idx] > 0.5:
                current = 0
            age[t, env_idx] = current
            current += 1
            if rollout.done[t, env_idx] > 0.5:
                current = 0
    return age


def _sample_list(items, max_items, seed):
    items = list(items)
    max_items = int(max_items)
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(items), size=max_items, replace=False)
    return [items[int(i)] for i in ids]


def _select_windows_by_step(valid, rollout, warmup, max_windows, windows_per_step, seed):
    if not valid:
        return [], np.zeros((0,), dtype=np.int32)
    age = _episode_age(rollout)
    by_step = defaultdict(list)
    for window in valid:
        t, env_idx = window
        step = int(age[int(t) + int(warmup), int(env_idx)])
        by_step[step].append(window)

    selected = []
    if int(windows_per_step) > 0:
        for step in sorted(by_step):
            step_seed = int(seed) + 17 * int(step)
            selected.extend(_sample_list(by_step[step], int(windows_per_step), step_seed))
    else:
        for step in sorted(by_step):
            selected.extend(by_step[step])

    selected = _sample_list(selected, int(max_windows), int(seed) + 100003)
    selected = sorted(selected)
    selected_steps = np.asarray([int(age[t + int(warmup), env_idx]) for t, env_idx in selected], dtype=np.int32)
    return selected, selected_steps


def _nan_array(rows, cols):
    out = np.full((int(rows), int(cols)), np.nan, dtype=np.float64)
    return out


def _safe_rmse(sq_sum, count):
    if count <= 0:
        return math.nan
    return math.sqrt(max(float(sq_sum) / float(count), 0.0))


def _mean_abs(abs_sum, count):
    if count <= 0:
        return math.nan
    return float(abs_sum) / float(count)


def _rolling_mean(y, window):
    window = int(window)
    arr = np.asarray(y, dtype=np.float64)
    if window <= 1 or arr.size <= 2:
        return arr
    out = np.empty_like(arr)
    half = max(window // 2, 1)
    for idx in range(arr.size):
        lo = max(idx - half, 0)
        hi = min(idx + half + 1, arr.size)
        chunk = arr[lo:hi]
        valid = np.isfinite(chunk)
        out[idx] = float(chunk[valid].mean()) if valid.any() else math.nan
    return out


def _run_obs_dim_predictions(args, world_model, rollout, windows, episode_steps):
    import torch

    warmup = int(args.model_warmup)
    horizon = int(args.model_horizon)
    obs_dim = int(rollout.obs.shape[-1])
    max_step = int(np.max(episode_steps)) if episode_steps.size else 0
    num_steps = max_step + 1

    h1_abs_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h1_sq_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h1_count = np.zeros(num_steps, dtype=np.float64)

    h5_endpoint_abs_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h5_endpoint_sq_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h5_endpoint_count = np.zeros(num_steps, dtype=np.float64)

    h5_mean_abs_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h5_mean_sq_sum = np.zeros((num_steps, obs_dim), dtype=np.float64)
    h5_mean_count = np.zeros(num_steps, dtype=np.float64)

    h1_overall_abs_sum = np.zeros(obs_dim, dtype=np.float64)
    h1_overall_sq_sum = np.zeros(obs_dim, dtype=np.float64)
    h1_overall_count = 0.0

    h5_endpoint_overall_abs_sum = np.zeros(obs_dim, dtype=np.float64)
    h5_endpoint_overall_sq_sum = np.zeros(obs_dim, dtype=np.float64)
    h5_endpoint_overall_count = 0.0

    h5_mean_overall_abs_sum = np.zeros(obs_dim, dtype=np.float64)
    h5_mean_overall_sq_sum = np.zeros(obs_dim, dtype=np.float64)
    h5_mean_overall_count = 0.0

    device = args.device
    batch_size = int(args.model_batch_size)
    world_model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            obs_warm, act_all, first_warm, targets = base_eval._batch_from_windows(
                rollout,
                windows,
                warmup,
                horizon,
                batch_start,
                batch_size,
                device,
            )
            bsz = int(obs_warm.shape[0])
            state = world_model.initial(bsz)
            for t in range(warmup + 1):
                _, state = world_model.get_inference_feat(state, obs_warm[:, t], first_warm[:, t])
                if t < warmup:
                    state = world_model.update_inference_state(state, act_all[:, t])

            pred_obs_steps = []
            for h in range(horizon):
                action = act_all[:, warmup + h]
                state = world_model.dynamic.img_step(state, action)
                stoch = world_model.dynamic.get_flatten_stoch(state)
                pred_obs = world_model.decode_obs(stoch) if hasattr(world_model, "decode_obs") else world_model.decoder(stoch)
                pred_obs_steps.append(pred_obs.float())

            pred_obs = torch.stack(pred_obs_steps, dim=1)
            obs_err = (pred_obs - targets["obs"]).detach().float()
            abs_err = obs_err.abs().cpu().numpy().astype(np.float64, copy=False)
            sq_err = obs_err.pow(2).cpu().numpy().astype(np.float64, copy=False)

            batch_steps = episode_steps[batch_start : batch_start + bsz]
            unique_steps = np.unique(batch_steps)
            for step in unique_steps.tolist():
                ids = np.nonzero(batch_steps == step)[0]
                if ids.size == 0:
                    continue
                h1_abs_sum[step] += abs_err[ids, 0, :].sum(axis=0)
                h1_sq_sum[step] += sq_err[ids, 0, :].sum(axis=0)
                h1_count[step] += float(ids.size)

                h5_endpoint_abs_sum[step] += abs_err[ids, horizon - 1, :].sum(axis=0)
                h5_endpoint_sq_sum[step] += sq_err[ids, horizon - 1, :].sum(axis=0)
                h5_endpoint_count[step] += float(ids.size)

                h5_mean_abs_sum[step] += abs_err[ids, :horizon, :].sum(axis=(0, 1))
                h5_mean_sq_sum[step] += sq_err[ids, :horizon, :].sum(axis=(0, 1))
                h5_mean_count[step] += float(ids.size * horizon)

            h1_overall_abs_sum += abs_err[:, 0, :].sum(axis=0)
            h1_overall_sq_sum += sq_err[:, 0, :].sum(axis=0)
            h1_overall_count += float(bsz)

            h5_endpoint_overall_abs_sum += abs_err[:, horizon - 1, :].sum(axis=0)
            h5_endpoint_overall_sq_sum += sq_err[:, horizon - 1, :].sum(axis=0)
            h5_endpoint_overall_count += float(bsz)

            h5_mean_overall_abs_sum += abs_err[:, :horizon, :].sum(axis=(0, 1))
            h5_mean_overall_sq_sum += sq_err[:, :horizon, :].sum(axis=(0, 1))
            h5_mean_overall_count += float(bsz * horizon)

    return {
        "num_steps": int(num_steps),
        "obs_dim": int(obs_dim),
        "h1_abs_sum": h1_abs_sum,
        "h1_sq_sum": h1_sq_sum,
        "h1_count": h1_count,
        "h5_endpoint_abs_sum": h5_endpoint_abs_sum,
        "h5_endpoint_sq_sum": h5_endpoint_sq_sum,
        "h5_endpoint_count": h5_endpoint_count,
        "h5_mean_abs_sum": h5_mean_abs_sum,
        "h5_mean_sq_sum": h5_mean_sq_sum,
        "h5_mean_count": h5_mean_count,
        "h1_overall_abs_sum": h1_overall_abs_sum,
        "h1_overall_sq_sum": h1_overall_sq_sum,
        "h1_overall_count": h1_overall_count,
        "h5_endpoint_overall_abs_sum": h5_endpoint_overall_abs_sum,
        "h5_endpoint_overall_sq_sum": h5_endpoint_overall_sq_sum,
        "h5_endpoint_overall_count": h5_endpoint_overall_count,
        "h5_mean_overall_abs_sum": h5_mean_overall_abs_sum,
        "h5_mean_overall_sq_sum": h5_mean_overall_sq_sum,
        "h5_mean_overall_count": h5_mean_overall_count,
    }


def _step_rows(policy, stats, prefix):
    abs_sum = stats[f"{prefix}_abs_sum"]
    sq_sum = stats[f"{prefix}_sq_sum"]
    count = stats[f"{prefix}_count"]
    rows = []
    for step in range(abs_sum.shape[0]):
        c = float(count[step])
        if c <= 0:
            continue
        for dim in range(abs_sum.shape[1]):
            rows.append(
                {
                    "policy": policy,
                    "episode_step": int(step),
                    "obs_dim": int(dim),
                    "count": int(c),
                    "mae": _mean_abs(abs_sum[step, dim], c),
                    "rmse": _safe_rmse(sq_sum[step, dim], c),
                }
            )
    return rows


def _overall_rows(policy, stats, prefix):
    abs_sum = stats[f"{prefix}_overall_abs_sum"]
    sq_sum = stats[f"{prefix}_overall_sq_sum"]
    count = float(stats[f"{prefix}_overall_count"])
    rows = []
    for dim in range(abs_sum.shape[0]):
        rows.append(
            {
                "policy": policy,
                "obs_dim": int(dim),
                "count": int(count),
                "mae": _mean_abs(abs_sum[dim], count),
                "rmse": _safe_rmse(sq_sum[dim], count),
            }
        )
    return rows


def _step_matrix(stats, prefix, metric="mae"):
    abs_sum = stats[f"{prefix}_abs_sum"]
    sq_sum = stats[f"{prefix}_sq_sum"]
    count = stats[f"{prefix}_count"]
    num_steps, obs_dim = abs_sum.shape
    matrix = _nan_array(num_steps, obs_dim)
    for step in range(num_steps):
        c = float(count[step])
        if c <= 0:
            continue
        if metric == "rmse":
            matrix[step] = np.sqrt(np.maximum(sq_sum[step] / c, 0.0))
        else:
            matrix[step] = abs_sum[step] / c
    return matrix


def _overall_vector(stats, prefix, metric="mae"):
    abs_sum = stats[f"{prefix}_overall_abs_sum"]
    sq_sum = stats[f"{prefix}_overall_sq_sum"]
    count = float(stats[f"{prefix}_overall_count"])
    if count <= 0:
        return np.full((abs_sum.shape[0],), np.nan, dtype=np.float64)
    if metric == "rmse":
        return np.sqrt(np.maximum(sq_sum / count, 0.0))
    return abs_sum / count


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_policy(save_dir, policy, stats, args):
    os.makedirs(save_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    plots = {}
    top_k = max(int(args.top_k_plot), 1)
    smooth = max(int(args.smooth_window), 1)
    steps = np.arange(int(stats["num_steps"]))
    h1_mae = _step_matrix(stats, "h1", metric="mae")
    h5_mean_mae = _step_matrix(stats, "h5_mean", metric="mae")
    h1_overall = _overall_vector(stats, "h1", metric="mae")
    h5_mean_overall = _overall_vector(stats, "h5_mean", metric="mae")
    h5_endpoint_overall = _overall_vector(stats, "h5_endpoint", metric="mae")

    top_h1 = np.argsort(np.nan_to_num(h1_overall, nan=-1.0))[-top_k:][::-1]
    top_h5 = np.argsort(np.nan_to_num(h5_mean_overall, nan=-1.0))[-top_k:][::-1]

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.95, h1_mae.shape[1]))
    for dim in range(h1_mae.shape[1]):
        ax.plot(steps, _rolling_mean(h1_mae[:, dim], smooth), lw=0.8, alpha=0.45, color=colors[dim])
    ax.set_title(f"{policy}: one-step obs-dim MAE curves")
    ax.set_xlabel("episode step at prediction start")
    ax.set_ylabel("h=1 MAE")
    path = os.path.join(save_dir, f"{policy}_obs_dim_h1_step_all_lines.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h1_step_all_lines"] = path

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for dim in top_h1.tolist():
        ax.plot(steps, _rolling_mean(h1_mae[:, dim], smooth), lw=1.6, label=f"obs_dim {dim}")
    ax.set_title(f"{policy}: top {len(top_h1)} one-step obs-dim MAE curves")
    ax.set_xlabel("episode step at prediction start")
    ax.set_ylabel("h=1 MAE")
    ax.legend(ncol=2, fontsize=8)
    path = os.path.join(save_dir, f"{policy}_obs_dim_h1_step_top_mae.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h1_step_top_mae"] = path

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    image = np.ma.masked_invalid(h1_mae.T)
    im = ax.imshow(image, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(f"{policy}: one-step obs-dim MAE heatmap")
    ax.set_xlabel("episode step at prediction start")
    ax.set_ylabel("obs dimension")
    fig.colorbar(im, ax=ax, label="h=1 MAE")
    path = os.path.join(save_dir, f"{policy}_obs_dim_h1_step_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h1_step_heatmap"] = path

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for dim in top_h5.tolist():
        ax.plot(steps, _rolling_mean(h5_mean_mae[:, dim], smooth), lw=1.6, label=f"obs_dim {dim}")
    ax.set_title(f"{policy}: top {len(top_h5)} five-step mean obs-dim MAE curves")
    ax.set_xlabel("episode step at prediction start")
    ax.set_ylabel("mean MAE over h=1..5")
    ax.legend(ncol=2, fontsize=8)
    path = os.path.join(save_dir, f"{policy}_obs_dim_h5_mean_step_top_mae.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h5_mean_step_top_mae"] = path

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    image = np.ma.masked_invalid(h5_mean_mae.T)
    im = ax.imshow(image, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(f"{policy}: five-step mean obs-dim MAE heatmap")
    ax.set_xlabel("episode step at prediction start")
    ax.set_ylabel("obs dimension")
    fig.colorbar(im, ax=ax, label="mean MAE over h=1..5")
    path = os.path.join(save_dir, f"{policy}_obs_dim_h5_mean_step_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h5_mean_step_heatmap"] = path

    x = np.arange(h1_overall.shape[0])
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    width = 0.28
    ax.bar(x - width, h1_overall, width, label="h=1")
    ax.bar(x, h5_mean_overall, width, label="mean h=1..5")
    ax.bar(x + width, h5_endpoint_overall, width, label="endpoint h=5")
    ax.set_title(f"{policy}: overall obs-dim prediction MAE")
    ax.set_xlabel("obs dimension")
    ax.set_ylabel("MAE")
    ax.legend()
    path = os.path.join(save_dir, f"{policy}_obs_dim_h1_h5_overall_bar.png")
    fig.savefig(path)
    plt.close(fig)
    plots[f"{policy}_obs_dim_h1_h5_overall_bar"] = path

    return plots


def _top_dim_summary(stats, prefix, top_k=10):
    vec = _overall_vector(stats, prefix, metric="mae")
    order = np.argsort(np.nan_to_num(vec, nan=-1.0))[::-1][: int(top_k)]
    return [{"obs_dim": int(dim), "mae": float(vec[dim])} for dim in order if np.isfinite(vec[dim])]


def _save_report(save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    step = result.get("checkpoint_step") or "unknown"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_dir, f"dfd_v5_wm_obs_dim_step_{step}_{stamp}.json")
    md_path = os.path.join(save_dir, f"dfd_v5_wm_obs_dim_step_{step}_{stamp}.md")
    json_safe = dict(result)
    json_safe.pop("_raw_policy_stats", None)
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(json_safe, fout, indent=2, ensure_ascii=False)

    lines = [
        "# DFD v5 World Model Obs-Dim Step Accuracy",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- config: `{result['config_path']}`",
        f"- checkpoint_step: `{result.get('checkpoint_step')}`",
        f"- env: `{result['env_name']}`",
        f"- policies: `{', '.join(result['policies'])}`",
        f"- num_envs/eval_episodes: `{result['num_envs']}/{result['eval_episodes']}`",
        f"- warmup/horizon: `{result['model_warmup']}/{result['model_horizon']}`",
        f"- step axis: `episode step at prediction start`",
        f"- h=5 mean: `mean per-dim error over open-loop h=1..5`",
        "",
        "## Window Counts",
        "",
        "| policy | valid windows | eval windows | episode steps | obs dim |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, item in result["policy_results"].items():
        lines.append(
            f"| {policy} | {item['num_valid_windows']} | {item['num_eval_windows']} | "
            f"{item['num_episode_steps']} | {item['obs_dim']} |"
        )

    lines += ["", "## Top Error Dims", ""]
    lines.append("| policy | rank | h1 top dim: MAE | h5 mean top dim: MAE |")
    lines.append("|---|---:|---|---|")
    for policy, item in result["policy_results"].items():
        h1 = item["top_h1_mae"]
        h5 = item["top_h5_mean_mae"]
        n = max(len(h1), len(h5))
        for idx in range(n):
            h1_txt = ""
            h5_txt = ""
            if idx < len(h1):
                h1_txt = f"dim {h1[idx]['obs_dim']}: {h1[idx]['mae']:.6f}"
            if idx < len(h5):
                h5_txt = f"dim {h5[idx]['obs_dim']}: {h5[idx]['mae']:.6f}"
            lines.append(f"| {policy} | {idx + 1} | {h1_txt} | {h5_txt} |")

    lines += ["", "## Plots", ""]
    for name, path in result.get("plots", {}).items():
        lines.append(f"- {name}: `{os.path.relpath(path, save_dir)}`")
    lines += ["", "## Tables", ""]
    for name, path in result.get("tables", {}).items():
        lines.append(f"- {name}: `{os.path.relpath(path, save_dir)}`")
    lines += ["", f"JSON report: `{json_path}`"]
    with open(md_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")
    return json_path, md_path


def _evaluate_policy(args, world_model, rollout, policy_index, save_dir):
    horizon = int(args.model_horizon)
    warmup = int(args.model_warmup)
    valid = base_eval._valid_model_windows(rollout, warmup, horizon)
    seed = int(args.seed or 0) + 3037 + int(policy_index) * 1009
    windows, steps = _select_windows_by_step(
        valid,
        rollout,
        warmup=warmup,
        max_windows=int(args.max_windows),
        windows_per_step=int(args.windows_per_step),
        seed=seed,
    )
    stats = _run_obs_dim_predictions(args, world_model, rollout, windows, steps)
    policy = rollout.policy

    h1_step_rows = _step_rows(policy, stats, "h1")
    h5_endpoint_step_rows = _step_rows(policy, stats, "h5_endpoint")
    h5_mean_step_rows = _step_rows(policy, stats, "h5_mean")
    h1_overall_rows = _overall_rows(policy, stats, "h1")
    h5_endpoint_overall_rows = _overall_rows(policy, stats, "h5_endpoint")
    h5_mean_overall_rows = _overall_rows(policy, stats, "h5_mean")

    table_paths = {
        f"{policy}_obs_dim_h1_by_step_csv": os.path.join(save_dir, f"{policy}_obs_dim_h1_by_step.csv"),
        f"{policy}_obs_dim_h5_endpoint_by_step_csv": os.path.join(
            save_dir, f"{policy}_obs_dim_h5_endpoint_by_step.csv"
        ),
        f"{policy}_obs_dim_h5_mean_by_step_csv": os.path.join(save_dir, f"{policy}_obs_dim_h5_mean_by_step.csv"),
        f"{policy}_obs_dim_h1_overall_csv": os.path.join(save_dir, f"{policy}_obs_dim_h1_overall.csv"),
        f"{policy}_obs_dim_h5_endpoint_overall_csv": os.path.join(
            save_dir, f"{policy}_obs_dim_h5_endpoint_overall.csv"
        ),
        f"{policy}_obs_dim_h5_mean_overall_csv": os.path.join(save_dir, f"{policy}_obs_dim_h5_mean_overall.csv"),
    }
    _write_csv(table_paths[f"{policy}_obs_dim_h1_by_step_csv"], h1_step_rows)
    _write_csv(table_paths[f"{policy}_obs_dim_h5_endpoint_by_step_csv"], h5_endpoint_step_rows)
    _write_csv(table_paths[f"{policy}_obs_dim_h5_mean_by_step_csv"], h5_mean_step_rows)
    _write_csv(table_paths[f"{policy}_obs_dim_h1_overall_csv"], h1_overall_rows)
    _write_csv(table_paths[f"{policy}_obs_dim_h5_endpoint_overall_csv"], h5_endpoint_overall_rows)
    _write_csv(table_paths[f"{policy}_obs_dim_h5_mean_overall_csv"], h5_mean_overall_rows)

    plot_paths = _plot_policy(save_dir, policy, stats, args)

    result = {
        "policy": policy,
        "num_valid_windows": int(len(valid)),
        "num_eval_windows": int(len(windows)),
        "num_episode_steps": int(stats["num_steps"]),
        "obs_dim": int(stats["obs_dim"]),
        "top_h1_mae": _top_dim_summary(stats, "h1", top_k=int(args.top_k_report)),
        "top_h5_mean_mae": _top_dim_summary(stats, "h5_mean", top_k=int(args.top_k_report)),
        "top_h5_endpoint_mae": _top_dim_summary(stats, "h5_endpoint", top_k=int(args.top_k_report)),
        "tables": table_paths,
        "plots": plot_paths,
    }
    return result, stats


def run_evaluation(args):
    import torch
    import torch.nn as nn

    from fdpi_reachability_dreamer.train import (
        _load_training_deps,
        _load_v4_full_checkpoint,
        build_agent,
        build_dual_policy,
        build_gd_critic,
        build_gp_critic,
        build_world_model,
        load_dfd_v5_config,
    )
    from fdpi_reachability_dreamer.train import build_env

    _load_training_deps()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False

    checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
        checkpoint_step = _checkpoint_step(checkpoint_path)
    else:
        checkpoint_step, checkpoint_path = _infer_latest_full_checkpoint(checkpoint_dir)
    config_path = base_eval._resolve_config_path(checkpoint_dir, args.config_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    conf = load_dfd_v5_config(config_path)
    conf = base_eval._set_eval_num_envs(conf, args.num_envs)
    if args.seed is not None:
        conf.defrost()
        conf.BasicSettings.Seed = int(args.seed)
        if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
            conf.Env.MakeKwargs.seed = int(args.seed)
        conf.freeze()
    torch.manual_seed(int(args.seed or conf.BasicSettings.Seed))
    np.random.seed(int(args.seed or conf.BasicSettings.Seed))

    first_env = build_env(args, conf)
    obs_dim = int(first_env.single_observation_space["policy"].shape[0])
    action_dim = int(first_env.single_action_space.shape[0])
    act = getattr(nn, conf.Models.Act)
    world_model = build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = build_agent(conf, action_dim, act, args.device)
    gp_critic = build_gp_critic(conf, action_dim, act, args.device)
    gd_critic = build_gd_critic(conf, action_dim, act, args.device)
    dual_policy = build_dual_policy(conf, action_dim, act, args.device)
    _load_v4_full_checkpoint(
        checkpoint_path,
        world_model=world_model,
        agent=agent,
        gp_critic=gp_critic,
        gd_critic=gd_critic,
        dual_policy=dual_policy,
        replay_buffer=None,
        device=args.device,
        load_optimizer=False,
        load_replay_buffer=False,
        load_rng=False,
    )
    world_model.eval()
    agent.eval()
    gp_critic.eval()
    gd_critic.eval()
    dual_policy.eval()
    modules = (world_model, agent, gp_critic, gd_critic, dual_policy)

    policies = [p.strip() for p in str(args.policies).split(",") if p.strip()]
    if args.smoke:
        policies = policies[:1]

    save_root = os.path.abspath(os.path.expanduser(args.save_dir))
    save_dir = os.path.join(save_root, f"dfd_v5_wm_obs_dim_step_{checkpoint_step or 'unknown'}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    policy_results = {}
    rollout_summaries = {}
    raw_stats = {}
    table_paths = {}
    plot_paths = {}

    for idx, policy in enumerate(policies):
        rollout_env = first_env if idx == 0 else None
        rollout_args = argparse.Namespace(**vars(args))
        rollout_args.smoke = False
        rollout = base_eval.collect_rollout(rollout_args, conf.clone(), modules, policy, idx, vec_env=rollout_env)
        if idx == 0:
            first_env = None
        rollout_summaries[policy] = rollout.summary
        policy_result, stats = _evaluate_policy(args, world_model, rollout, idx, save_dir)
        policy_results[policy] = policy_result
        raw_stats[policy] = stats
        table_paths.update(policy_result.get("tables", {}))
        plot_paths.update(policy_result.get("plots", {}))

    result = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "config_path": config_path,
        "env_name": args.env_name,
        "device": args.device,
        "seed": int(args.seed or conf.BasicSettings.Seed),
        "num_envs": int(args.num_envs),
        "eval_steps": int(args.eval_steps),
        "eval_episodes": int(args.eval_episodes) if args.eval_episodes is not None else None,
        "greedy": bool(args.greedy),
        "policies": policies,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "model_warmup": int(args.model_warmup),
        "model_horizon": int(args.model_horizon),
        "max_windows": int(args.max_windows),
        "windows_per_step": int(args.windows_per_step),
        "rollout_summaries": rollout_summaries,
        "policy_results": policy_results,
        "tables": table_paths,
        "plots": plot_paths,
        "_raw_policy_stats": raw_stats,
    }
    json_path, md_path = _save_report(save_dir, result)

    print("\nDFD v5 world-model obs-dim step accuracy")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config: {config_path}")
    print(f"  save_dir: {save_dir}")
    for policy, item in policy_results.items():
        print(
            f"  {policy}: valid_windows={item['num_valid_windows']}, "
            f"eval_windows={item['num_eval_windows']}, "
            f"episode_steps={item['num_episode_steps']}, obs_dim={item['obs_dim']}"
        )
    print(f"  json: {json_path}")
    print(f"  md: {md_path}")
    return result


def main():
    warnings.filterwarnings("ignore")
    default_ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt", "dfd-v5-task-first-from-750k", "20260603_121444")
    parser = argparse.ArgumentParser(description="Per-observation-dimension step-wise DFD v5 world-model accuracy.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1")
    parser.add_argument("--device", "-device", type=str, default="cuda:0")
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=98304)
    parser.add_argument("--eval_episodes", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--policies", type=str, default="main")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--model_warmup", type=int, default=16)
    parser.add_argument("--model_horizon", type=int, default=5)
    parser.add_argument("--model_batch_size", type=int, default=512)
    parser.add_argument("--max_windows", type=int, default=0, help="0 means use all valid windows.")
    parser.add_argument("--windows_per_step", type=int, default=0, help="0 means no per-step subsampling.")
    parser.add_argument("--top_k_plot", type=int, default=12)
    parser.add_argument("--top_k_report", type=int, default=10)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_wm_obs_dim_step"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.greedy = not args.stochastic
    if int(args.model_horizon) != 5:
        raise ValueError("This script is intended for five-step obs-dim evaluation; keep --model_horizon 5.")

    _launch_isaac(headless=not args.render)
    try:
        run_evaluation(args)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
