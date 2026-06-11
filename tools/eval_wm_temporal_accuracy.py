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


def _checkpoint_step(path):
    match = re.search(r"full_state(?:_v5|_v4)?_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _binary_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos <= 0 or neg <= 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    rank_sum_pos = full_ranks[labels].sum()
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _summary(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _sample_list(items, max_items, seed):
    items = list(items)
    max_items = int(max_items)
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(items), size=max_items, replace=False)
    return [items[int(i)] for i in ids]


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


def _window_metadata(rollout, windows, warmup, horizon):
    age = _episode_age(rollout)
    rows = []
    total = int(rollout.done.shape[0])
    for row, (t, env_idx) in enumerate(windows):
        start = int(t) + int(warmup)
        stop = start + int(horizon)
        force_seq = rollout.cost_force[start:stop, env_idx]
        binary_seq = rollout.binary_cost[start:stop, env_idx]
        extreme_seq = rollout.extreme_cost[start:stop, env_idx]
        rows.append(
            {
                "row": int(row),
                "t": int(t),
                "env_idx": int(env_idx),
                "pred_start": int(start),
                "episode_age": int(age[start, env_idx]),
                "rollout_frac": float(start / max(total - 1, 1)),
                "start_force": float(rollout.cost_force[start, env_idx]),
                "max_force_h15": float(np.max(force_seq)) if force_seq.size else 0.0,
                "mean_force_h15": float(np.mean(force_seq)) if force_seq.size else 0.0,
                "any_cost_h15": bool(np.any(binary_seq > 0.5)),
                "any_extreme_h15": bool(np.any(extreme_seq > 0.5)),
            }
        )
    return rows


def _group_masks(metadata, near_force_threshold, cost_force_threshold):
    n = len(metadata)
    age = np.asarray([m["episode_age"] for m in metadata], dtype=np.float64)
    frac = np.asarray([m["rollout_frac"] for m in metadata], dtype=np.float64)
    start_force = np.asarray([m["start_force"] for m in metadata], dtype=np.float64)
    max_force = np.asarray([m["max_force_h15"] for m in metadata], dtype=np.float64)
    any_cost = np.asarray([m["any_cost_h15"] for m in metadata], dtype=bool)
    any_extreme = np.asarray([m["any_extreme_h15"] for m in metadata], dtype=bool)

    masks = {
        "all": np.ones(n, dtype=bool),
        "episode_age_0_49": age < 50,
        "episode_age_50_119": (age >= 50) & (age < 120),
        "episode_age_120_plus": age >= 120,
        "rollout_q1": frac < 0.25,
        "rollout_q2": (frac >= 0.25) & (frac < 0.50),
        "rollout_q3": (frac >= 0.50) & (frac < 0.75),
        "rollout_q4": frac >= 0.75,
        "bottom_near_start": start_force > float(near_force_threshold),
        "bottom_near_h15": max_force > float(near_force_threshold),
        "cost_event_h15": any_cost,
        "extreme_event_h15": any_extreme,
        "bottom_clear_h15": max_force <= float(near_force_threshold),
        "bottom_over_cost_threshold_h15": max_force > float(cost_force_threshold),
    }
    return masks


def _select_windows_for_groups(valid, rollout, args, policy_index):
    warmup = int(args.model_warmup)
    horizon = int(args.model_horizon)
    seed = int(args.seed or 0) + 2000 + policy_index * 101
    base_windows = _sample_list(valid, int(args.model_eval_windows), seed)
    all_meta = _window_metadata(rollout, valid, warmup, horizon)
    masks = _group_masks(all_meta, args.near_bottom_force_threshold, args.cost_force_threshold)

    selected = set(base_windows)
    for group_name in (
        "episode_age_0_49",
        "episode_age_50_119",
        "episode_age_120_plus",
        "bottom_near_start",
        "bottom_near_h15",
        "cost_event_h15",
        "extreme_event_h15",
        "bottom_over_cost_threshold_h15",
    ):
        candidate_ids = np.nonzero(masks[group_name])[0].tolist()
        candidates = [valid[i] for i in candidate_ids]
        group_seed = seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(group_name))
        for item in _sample_list(candidates, int(args.group_eval_windows), group_seed):
            selected.add(item)
    selected = sorted(selected)
    return selected


def _run_world_model_predictions(args, world_model, rollout, windows):
    import torch

    warmup = int(args.model_warmup)
    horizon = int(args.model_horizon)
    n = len(windows)
    obs_abs_mean = np.empty((n, horizon), dtype=np.float32)
    obs_sq_mean = np.empty((n, horizon), dtype=np.float32)
    reward_abs = np.empty((n, horizon), dtype=np.float32)
    reward_sq = np.empty((n, horizon), dtype=np.float32)
    cost_abs = np.empty((n, horizon), dtype=np.float32)
    cost_sq = np.empty((n, horizon), dtype=np.float32)
    pred_cost = np.empty((n, horizon), dtype=np.float32)
    true_cost = np.empty((n, horizon), dtype=np.float32)
    true_binary = np.empty((n, horizon), dtype=np.float32)
    pred_extreme = np.empty((n, horizon), dtype=np.float32)
    true_extreme = np.empty((n, horizon), dtype=np.float32)

    device = args.device
    batch_size = int(args.model_batch_size)
    world_model.eval()
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
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
            pred_reward_steps = []
            pred_cost_steps = []
            pred_extreme_steps = []
            for h in range(horizon):
                action = act_all[:, warmup + h]
                state = world_model.dynamic.img_step(state, action)
                stoch = world_model.dynamic.get_flatten_stoch(state)
                feat = world_model.dynamic.get_feat(state)
                pred_obs = world_model.decode_obs(stoch).float() if hasattr(world_model, "decode_obs") else world_model.decoder(stoch).float()
                pred_reward = world_model.twohot_loss.decode(world_model.reward_head(state["deter"])).float()
                if hasattr(world_model, "predict_cost"):
                    cost, extreme, _ = world_model.predict_cost(feat)
                    cost = cost.float()
                    extreme = extreme.float()
                else:
                    cost = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
                    extreme = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
                pred_obs_steps.append(pred_obs)
                pred_reward_steps.append(pred_reward.reshape(bsz, 1))
                pred_cost_steps.append(cost.reshape(bsz, 1))
                pred_extreme_steps.append(extreme.reshape(bsz, 1))

            pred_obs = torch.stack(pred_obs_steps, dim=1)
            pred_reward = torch.stack(pred_reward_steps, dim=1)
            pred_cost_batch = torch.stack(pred_cost_steps, dim=1)
            pred_extreme_batch = torch.stack(pred_extreme_steps, dim=1)
            obs_err = (pred_obs - targets["obs"]).detach().float()
            reward_err = (pred_reward - targets["reward"]).detach().float()
            cost_err = (pred_cost_batch - targets["cost"]).detach().float()
            sl = slice(batch_start, batch_start + bsz)
            obs_abs_mean[sl] = obs_err.abs().mean(dim=-1).cpu().numpy()
            obs_sq_mean[sl] = obs_err.pow(2).mean(dim=-1).cpu().numpy()
            reward_abs[sl] = reward_err.abs().reshape(bsz, horizon).cpu().numpy()
            reward_sq[sl] = reward_err.pow(2).reshape(bsz, horizon).cpu().numpy()
            cost_abs[sl] = cost_err.abs().reshape(bsz, horizon).cpu().numpy()
            cost_sq[sl] = cost_err.pow(2).reshape(bsz, horizon).cpu().numpy()
            pred_cost[sl] = pred_cost_batch.reshape(bsz, horizon).cpu().numpy()
            true_cost[sl] = targets["cost"].reshape(bsz, horizon).cpu().numpy()
            true_binary[sl] = targets["binary_cost"].reshape(bsz, horizon).cpu().numpy()
            pred_extreme[sl] = pred_extreme_batch.reshape(bsz, horizon).cpu().numpy()
            true_extreme[sl] = targets["extreme_cost"].reshape(bsz, horizon).cpu().numpy()

    return {
        "obs_abs_mean": obs_abs_mean,
        "obs_sq_mean": obs_sq_mean,
        "reward_abs": reward_abs,
        "reward_sq": reward_sq,
        "cost_abs": cost_abs,
        "cost_sq": cost_sq,
        "pred_cost": pred_cost,
        "true_cost": true_cost,
        "true_binary": true_binary,
        "pred_extreme": pred_extreme,
        "true_extreme": true_extreme,
    }


def _aggregate_group(policy, group_name, mask, predictions, metadata, horizons):
    ids = np.nonzero(mask)[0]
    rows = []
    max_force = np.asarray([m["max_force_h15"] for m in metadata], dtype=np.float64)
    start_force = np.asarray([m["start_force"] for m in metadata], dtype=np.float64)
    for h_idx, horizon in enumerate(horizons):
        if ids.size == 0:
            rows.append(
                {
                    "policy": policy,
                    "group": group_name,
                    "horizon": int(horizon),
                    "count": 0,
                    "obs_mae": None,
                    "obs_rmse": None,
                    "reward_mae": None,
                    "reward_rmse": None,
                    "cost_mae": None,
                    "cost_rmse": None,
                    "cost_event_rate": None,
                    "cost_auc": None,
                    "pred_cost_mean": None,
                    "true_cost_mean": None,
                    "start_force_mean": None,
                    "max_force_h15_mean": None,
                }
            )
            continue
        obs_mae = float(np.mean(predictions["obs_abs_mean"][ids, h_idx]))
        obs_rmse = float(math.sqrt(float(np.mean(predictions["obs_sq_mean"][ids, h_idx]))))
        reward_mae = float(np.mean(predictions["reward_abs"][ids, h_idx]))
        reward_rmse = float(math.sqrt(float(np.mean(predictions["reward_sq"][ids, h_idx]))))
        cost_mae = float(np.mean(predictions["cost_abs"][ids, h_idx]))
        cost_rmse = float(math.sqrt(float(np.mean(predictions["cost_sq"][ids, h_idx]))))
        pred = predictions["pred_cost"][ids, h_idx]
        target_binary = predictions["true_binary"][ids, h_idx] > 0.5
        rows.append(
            {
                "policy": policy,
                "group": group_name,
                "horizon": int(horizon),
                "count": int(ids.size),
                "obs_mae": obs_mae,
                "obs_rmse": obs_rmse,
                "reward_mae": reward_mae,
                "reward_rmse": reward_rmse,
                "cost_mae": cost_mae,
                "cost_rmse": cost_rmse,
                "cost_event_rate": float(target_binary.mean()) if target_binary.size else None,
                "cost_auc": _binary_auc(pred, target_binary),
                "pred_cost_mean": float(np.mean(pred)),
                "true_cost_mean": float(np.mean(predictions["true_cost"][ids, h_idx])),
                "start_force_mean": float(np.mean(start_force[ids])),
                "max_force_h15_mean": float(np.mean(max_force[ids])),
            }
        )
    return rows


def _evaluate_policy_world_model(args, world_model, rollout, policy_index):
    horizon = int(args.model_horizon)
    warmup = int(args.model_warmup)
    valid = base_eval._valid_model_windows(rollout, warmup, horizon)
    if not valid:
        return {
            "policy": rollout.policy,
            "num_valid_windows": 0,
            "error": "no valid non-terminal windows",
        }
    windows = _select_windows_for_groups(valid, rollout, args, policy_index)
    metadata = _window_metadata(rollout, windows, warmup, horizon)
    predictions = _run_world_model_predictions(args, world_model, rollout, windows)
    masks = _group_masks(metadata, args.near_bottom_force_threshold, args.cost_force_threshold)
    horizons = list(range(1, horizon + 1))
    rows = []
    for group_name, mask in masks.items():
        rows.extend(_aggregate_group(rollout.policy, group_name, mask, predictions, metadata, horizons))

    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    selected_horizon_rows = [
        row
        for row in rows
        if int(row["horizon"]) in {1, 5, 15}
        and row["group"]
        in {
            "all",
            "episode_age_0_49",
            "episode_age_50_119",
            "episode_age_120_plus",
            "bottom_near_h15",
            "cost_event_h15",
            "bottom_clear_h15",
        }
    ]
    return {
        "policy": rollout.policy,
        "num_valid_windows": int(len(valid)),
        "num_eval_windows": int(len(windows)),
        "warmup": int(warmup),
        "horizon": int(horizon),
        "near_bottom_force_threshold": float(args.near_bottom_force_threshold),
        "cost_force_threshold": float(args.cost_force_threshold),
        "group_counts": counts,
        "horizon_metrics": rows,
        "selected_horizon_metrics": selected_horizon_rows,
        "metadata_summary": {
            "episode_age": _summary([m["episode_age"] for m in metadata]),
            "start_force": _summary([m["start_force"] for m in metadata]),
            "max_force_h15": _summary([m["max_force_h15"] for m in metadata]),
        },
    }


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_map(rows):
    return {
        (row["policy"], row["group"], int(row["horizon"])): row
        for row in rows
    }


def _plot_results(save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    rows = result["tables_data"]["horizon_metrics"]
    metric_map = _metric_map(rows)
    plots = {}
    policies = result["policies"]
    horizon = int(result["model_horizon"])
    x = np.arange(1, horizon + 1)

    def series(policy, group, metric):
        values = []
        for h in x:
            row = metric_map.get((policy, group, int(h)), {})
            value = row.get(metric)
            values.append(np.nan if value is None else float(value))
        return values

    for policy in policies:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for group in ("all", "bottom_near_h15", "cost_event_h15", "bottom_clear_h15"):
            if metric_map.get((policy, group, 1), {}).get("count", 0) <= 0:
                continue
            label = f"{group} (n={metric_map[(policy, group, 1)]['count']})"
            axes[0].plot(x, series(policy, group, "obs_rmse"), marker="o", markersize=3, label=label)
            axes[1].plot(x, series(policy, group, "cost_mae"), marker="o", markersize=3, label=label)
            axes[2].plot(x, series(policy, group, "reward_rmse"), marker="o", markersize=3, label=label)
        axes[0].set_title(f"{policy}: observation RMSE")
        axes[1].set_title(f"{policy}: cost MAE")
        axes[2].set_title(f"{policy}: reward RMSE")
        for ax in axes:
            ax.set_xlabel("open-loop horizon")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        path = os.path.join(save_dir, f"{policy}_bottom_region_horizon_errors.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[f"{policy}_bottom_region_horizon_errors"] = path

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for group in ("episode_age_0_49", "episode_age_50_119", "episode_age_120_plus"):
            if metric_map.get((policy, group, 1), {}).get("count", 0) <= 0:
                continue
            label = f"{group} (n={metric_map[(policy, group, 1)]['count']})"
            axes[0].plot(x, series(policy, group, "obs_rmse"), marker="o", markersize=3, label=label)
            axes[1].plot(x, series(policy, group, "cost_mae"), marker="o", markersize=3, label=label)
        axes[0].set_title(f"{policy}: obs RMSE by episode age")
        axes[1].set_title(f"{policy}: cost MAE by episode age")
        for ax in axes:
            ax.set_xlabel("open-loop horizon")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        path = os.path.join(save_dir, f"{policy}_episode_age_horizon_errors.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[f"{policy}_episode_age_horizon_errors"] = path

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for group in ("rollout_q1", "rollout_q2", "rollout_q3", "rollout_q4"):
            if metric_map.get((policy, group, 1), {}).get("count", 0) <= 0:
                continue
            label = f"{group} (n={metric_map[(policy, group, 1)]['count']})"
            axes[0].plot(x, series(policy, group, "obs_rmse"), marker="o", markersize=3, label=label)
            axes[1].plot(x, series(policy, group, "cost_mae"), marker="o", markersize=3, label=label)
        axes[0].set_title(f"{policy}: obs RMSE by rollout time")
        axes[1].set_title(f"{policy}: cost MAE by rollout time")
        for ax in axes:
            ax.set_xlabel("open-loop horizon")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        path = os.path.join(save_dir, f"{policy}_rollout_time_horizon_errors.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[f"{policy}_rollout_time_horizon_errors"] = path

        selected_groups = ("all", "bottom_near_h15", "cost_event_h15", "bottom_clear_h15")
        selected_h = (1, 5, 15)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        width = 0.18
        gx = np.arange(len(selected_groups))
        for idx, h in enumerate(selected_h):
            axes[0].bar(
                gx + (idx - 1) * width,
                [metric_map.get((policy, g, h), {}).get("obs_rmse", np.nan) for g in selected_groups],
                width,
                label=f"h={h}",
            )
            axes[1].bar(
                gx + (idx - 1) * width,
                [metric_map.get((policy, g, h), {}).get("cost_mae", np.nan) for g in selected_groups],
                width,
                label=f"h={h}",
            )
        axes[0].set_title(f"{policy}: selected horizon obs RMSE")
        axes[1].set_title(f"{policy}: selected horizon cost MAE")
        for ax in axes:
            ax.set_xticks(gx, selected_groups, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
        path = os.path.join(save_dir, f"{policy}_h1_h5_h15_bars.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[f"{policy}_h1_h5_h15_bars"] = path

    return plots


def _fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _save_report(save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    step = result.get("checkpoint_step") or "unknown"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_dir, f"dfd_v5_wm_temporal_{step}_{stamp}.json")
    md_path = os.path.join(save_dir, f"dfd_v5_wm_temporal_{step}_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

    selected_rows = result["tables_data"]["selected_horizon_metrics"]
    lines = [
        "# DFD v5 World Model Temporal Accuracy",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- config: `{result['config_path']}`",
        f"- checkpoint_step: `{result.get('checkpoint_step')}`",
        f"- env: `{result['env_name']}`",
        f"- policies: `{', '.join(result['policies'])}`",
        f"- num_envs/eval_episodes: `{result['num_envs']}/{result['eval_episodes']}`",
        f"- warmup/horizon: `{result['model_warmup']}/{result['model_horizon']}`",
        f"- selected eval windows: `{result['selected_eval_windows']}`",
        f"- near_bottom_force_threshold: `{result['near_bottom_force_threshold']}`",
        "",
        "## Group Counts",
        "",
        "| policy | valid windows | eval windows | all | age 0-49 | age 50-119 | age 120+ | bottom near h15 | cost event h15 | extreme event h15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, item in result["policy_results"].items():
        counts = item.get("group_counts", {})
        lines.append(
            f"| {policy} | {item.get('num_valid_windows')} | {item.get('num_eval_windows')} | "
            f"{counts.get('all', 0)} | {counts.get('episode_age_0_49', 0)} | "
            f"{counts.get('episode_age_50_119', 0)} | {counts.get('episode_age_120_plus', 0)} | "
            f"{counts.get('bottom_near_h15', 0)} | {counts.get('cost_event_h15', 0)} | "
            f"{counts.get('extreme_event_h15', 0)} |"
        )
    lines += [
        "",
        "## H=1/5/15 Metrics",
        "",
        "| policy | group | h | count | obs RMSE | obs MAE | reward RMSE | cost MAE | cost RMSE | cost event rate | cost AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['policy']} | {row['group']} | {row['horizon']} | {row['count']} | "
            f"{_fmt(row.get('obs_rmse'))} | {_fmt(row.get('obs_mae'))} | {_fmt(row.get('reward_rmse'))} | "
            f"{_fmt(row.get('cost_mae'), 6)} | {_fmt(row.get('cost_rmse'), 6)} | "
            f"{_fmt(row.get('cost_event_rate'), 4)} | {_fmt(row.get('cost_auc'), 4)} |"
        )
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

    policy_results = {}
    rollout_summaries = {}
    for idx, policy in enumerate(policies):
        rollout_env = first_env if idx == 0 else None
        rollout_args = argparse.Namespace(**vars(args))
        rollout_args.smoke = False
        rollout = base_eval.collect_rollout(rollout_args, conf.clone(), modules, policy, idx, vec_env=rollout_env)
        if idx == 0:
            first_env = None
        rollout_summaries[policy] = rollout.summary
        policy_results[policy] = _evaluate_policy_world_model(args, world_model, rollout, idx)

    save_root = os.path.abspath(os.path.expanduser(args.save_dir))
    save_dir = os.path.join(save_root, f"dfd_v5_wm_temporal_{checkpoint_step or 'unknown'}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    horizon_rows = []
    selected_rows = []
    for item in policy_results.values():
        horizon_rows.extend(item.get("horizon_metrics", []))
        selected_rows.extend(item.get("selected_horizon_metrics", []))
    horizon_csv = os.path.join(save_dir, "wm_temporal_horizon_metrics.csv")
    selected_csv = os.path.join(save_dir, "wm_temporal_h1_h5_h15_metrics.csv")
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(selected_csv, selected_rows)

    result = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "config_path": config_path,
        "env_name": args.env_name,
        "device": args.device,
        "seed": int(args.seed or conf.BasicSettings.Seed),
        "num_envs": int(args.num_envs),
        "eval_episodes": int(args.eval_episodes) if args.eval_episodes is not None else None,
        "greedy": bool(args.greedy),
        "policies": policies,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "model_warmup": int(args.model_warmup),
        "model_horizon": int(args.model_horizon),
        "selected_eval_windows": int(args.model_eval_windows),
        "group_eval_windows": int(args.group_eval_windows),
        "near_bottom_force_threshold": float(args.near_bottom_force_threshold),
        "cost_force_threshold": float(args.cost_force_threshold),
        "cost_config": base_eval._cfg_to_dict(conf.FDPIRegimeDreamer.ContinuousCost),
        "rollout_summaries": rollout_summaries,
        "policy_results": policy_results,
        "tables": {
            "wm_temporal_horizon_metrics_csv": horizon_csv,
            "wm_temporal_h1_h5_h15_metrics_csv": selected_csv,
        },
        "tables_data": {
            "horizon_metrics": horizon_rows,
            "selected_horizon_metrics": selected_rows,
        },
    }
    result["plots"] = _plot_results(save_dir, result)
    json_path, md_path = _save_report(save_dir, result)

    print("\nDFD v5 world-model temporal accuracy")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config: {config_path}")
    print(f"  save_dir: {save_dir}")
    for policy, item in policy_results.items():
        counts = item.get("group_counts", {})
        print(
            f"  {policy}: valid_windows={item.get('num_valid_windows')}, "
            f"eval_windows={item.get('num_eval_windows')}, "
            f"bottom_near_h15={counts.get('bottom_near_h15')}, "
            f"cost_event_h15={counts.get('cost_event_h15')}"
        )
    print(f"  json: {json_path}")
    print(f"  md: {md_path}")
    return result


def main():
    warnings.filterwarnings("ignore")
    default_ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt", "dfd-v5-task-first-from-750k", "20260603_121444")
    parser = argparse.ArgumentParser(description="Temporal and bottom-region DFD v5 world-model prediction evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=81920)
    parser.add_argument("--eval_episodes", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--policies", type=str, default="main")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--model_warmup", type=int, default=16)
    parser.add_argument("--model_horizon", type=int, default=15)
    parser.add_argument("--model_eval_windows", type=int, default=8192)
    parser.add_argument("--group_eval_windows", type=int, default=4096)
    parser.add_argument("--model_batch_size", type=int, default=512)
    parser.add_argument("--near_bottom_force_threshold", type=float, default=0.02)
    parser.add_argument("--cost_force_threshold", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_wm_temporal"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.greedy = not args.stochastic

    _launch_isaac(headless=not args.render)
    try:
        run_evaluation(args)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
