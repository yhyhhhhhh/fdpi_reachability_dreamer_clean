from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


OBS_DIM_NAMES = [
    "joint_pos_0",
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "joint_pos_7",
    "joint_vel_0",
    "joint_vel_1",
    "joint_vel_2",
    "joint_vel_3",
    "joint_vel_4",
    "joint_vel_5",
    "joint_vel_6",
    "joint_vel_7",
    "gripper_q_0",
    "gripper_q_1",
    "gripper_open",
    "gripper_closed",
    "object_local_x",
    "object_local_y",
    "object_local_z",
    "object_minus_finger_x",
    "object_minus_finger_y",
    "object_minus_finger_z",
    "object_minus_blade_mid_x",
    "object_minus_blade_mid_y",
    "object_minus_blade_mid_z",
    "object_dist",
    "blade1_local_x",
    "blade1_local_y",
    "blade1_local_z",
    "blade2_local_x",
    "blade2_local_y",
    "blade2_local_z",
    "blade_dist",
    "lift_height",
    "object_to_goal_local_x",
    "object_to_goal_local_y",
    "object_to_goal_local_z",
    "goal_dist",
    "is_lifted_obs",
    "left_force",
    "right_force",
    "contact_strength",
    "bilateral_contact",
]


def _checkpoint_step(path):
    import re

    match = re.search(r"full_state(?:_v5|_v4)?_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _dim_name(dim, obs_dim):
    if obs_dim == len(OBS_DIM_NAMES):
        return OBS_DIM_NAMES[dim]
    return f"obs_{dim}"


def _dim_group(name):
    if name.startswith("joint_pos"):
        return "joint_pos"
    if name.startswith("joint_vel"):
        return "joint_vel"
    if name.startswith("gripper"):
        return "gripper"
    if name.startswith("object_") or name in {"goal_dist", "lift_height"}:
        return "task_geometry"
    if name.startswith("blade"):
        return "blade_geometry"
    if "force" in name or name in {"contact_strength", "bilateral_contact"}:
        return "contact"
    if name == "is_lifted_obs":
        return "progress_flag"
    return "other"


def _flatten_obs_arrays(rollout, include_next_obs=True):
    arrays = [np.asarray(rollout.obs, dtype=np.float32).reshape(-1, rollout.obs.shape[-1])]
    if include_next_obs:
        arrays.append(np.asarray(rollout.next_obs, dtype=np.float32).reshape(-1, rollout.next_obs.shape[-1]))
    return np.concatenate(arrays, axis=0)


def _stats_from_array(array, std_floor):
    array = np.asarray(array, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    raw_std = np.nanstd(array, axis=0)
    min_value = np.nanmin(array, axis=0)
    max_value = np.nanmax(array, axis=0)
    p01 = np.nanpercentile(array, 1.0, axis=0)
    p05 = np.nanpercentile(array, 5.0, axis=0)
    p50 = np.nanpercentile(array, 50.0, axis=0)
    p95 = np.nanpercentile(array, 95.0, axis=0)
    p99 = np.nanpercentile(array, 99.0, axis=0)
    std = np.maximum(raw_std, float(std_floor))
    return {
        "mean": mean.astype(np.float32),
        "raw_std": raw_std.astype(np.float32),
        "std": std.astype(np.float32),
        "min": min_value.astype(np.float32),
        "max": max_value.astype(np.float32),
        "p01": p01.astype(np.float32),
        "p05": p05.astype(np.float32),
        "p50": p50.astype(np.float32),
        "p95": p95.astype(np.float32),
        "p99": p99.astype(np.float32),
    }


def _write_csv(path, stats, obs_dim, std_floor):
    fieldnames = [
        "obs_dim",
        "name",
        "group",
        "mean",
        "raw_std",
        "std_for_norm",
        "std_floor",
        "min",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "max",
        "p95_p05",
        "range",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for dim in range(obs_dim):
            name = _dim_name(dim, obs_dim)
            writer.writerow(
                {
                    "obs_dim": dim,
                    "name": name,
                    "group": _dim_group(name),
                    "mean": float(stats["mean"][dim]),
                    "raw_std": float(stats["raw_std"][dim]),
                    "std_for_norm": float(stats["std"][dim]),
                    "std_floor": float(std_floor),
                    "min": float(stats["min"][dim]),
                    "p01": float(stats["p01"][dim]),
                    "p05": float(stats["p05"][dim]),
                    "p50": float(stats["p50"][dim]),
                    "p95": float(stats["p95"][dim]),
                    "p99": float(stats["p99"][dim]),
                    "max": float(stats["max"][dim]),
                    "p95_p05": float(stats["p95"][dim] - stats["p05"][dim]),
                    "range": float(stats["max"][dim] - stats["min"][dim]),
                }
            )


def _write_json(path, result):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with open(path, "w", encoding="utf-8") as fout:
        json.dump(convert(result), fout, indent=2, ensure_ascii=False)


def _plot_stats(save_dir, stats, obs_dim, std_floor):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    dims = np.arange(obs_dim)
    names = [_dim_name(dim, obs_dim) for dim in range(obs_dim)]
    plot_paths = {}

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    ax.bar(dims, stats["raw_std"], width=0.8, color="#3c7d87")
    ax.axhline(float(std_floor), color="#b23a48", linestyle="--", linewidth=1.3, label=f"std floor={std_floor:g}")
    ax.set_yscale("log")
    ax.set_title("Observation per-dimension raw std")
    ax.set_xlabel("obs dim")
    ax.set_ylabel("raw std (log)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    path = os.path.join(save_dir, "obs_dim_raw_std.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plot_paths["raw_std"] = path

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    p95_p05 = stats["p95"] - stats["p05"]
    ax.bar(dims, p95_p05, width=0.8, color="#7a5c58")
    ax.set_title("Observation per-dimension p95-p05 range")
    ax.set_xlabel("obs dim")
    ax.set_ylabel("p95 - p05")
    ax.grid(alpha=0.25, axis="y")
    path = os.path.join(save_dir, "obs_dim_p95_p05_range.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plot_paths["p95_p05_range"] = path

    top_idx = np.argsort(-stats["raw_std"])[: min(16, obs_dim)]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(top_idx.size)
    ax.bar(x, stats["raw_std"][top_idx], width=0.68, color="#5b7f95")
    ax.set_xticks(x, [f"{idx}:{names[idx]}" for idx in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_title("Largest raw-std observation dimensions")
    ax.set_ylabel("raw std")
    ax.grid(alpha=0.25, axis="y")
    path = os.path.join(save_dir, "obs_dim_top_raw_std.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plot_paths["top_raw_std"] = path

    top_range_idx = np.argsort(-p95_p05)[: min(16, obs_dim)]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(top_range_idx.size)
    ax.bar(x, p95_p05[top_range_idx], width=0.68, color="#886f2f")
    ax.set_xticks(x, [f"{idx}:{names[idx]}" for idx in top_range_idx], rotation=45, ha="right", fontsize=8)
    ax.set_title("Largest p95-p05 observation dimensions")
    ax.set_ylabel("p95 - p05")
    ax.grid(alpha=0.25, axis="y")
    path = os.path.join(save_dir, "obs_dim_top_p95_p05_range.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plot_paths["top_p95_p05_range"] = path

    return plot_paths


def _write_report(path, result, stats, obs_dim, save_dir):
    top_std = np.argsort(-stats["raw_std"])[: min(12, obs_dim)]
    top_range = np.argsort(-(stats["p95"] - stats["p05"]))[: min(12, obs_dim)]
    floored = np.where(stats["raw_std"] < float(result["std_floor"]))[0]
    lines = [
        "# DFD v5 Observation Normalizer",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- checkpoint_step: `{result['checkpoint_step']}`",
        f"- config: `{result['config_path']}`",
        f"- policies: `{', '.join(result['policies'])}`",
        f"- num_envs: `{result['num_envs']}`",
        f"- eval_steps_per_policy: `{result['eval_steps']}`",
        f"- eval_episodes_per_policy: `{result['eval_episodes']}`",
        f"- include_next_obs: `{result['include_next_obs']}`",
        f"- sample_count: `{result['sample_count']}`",
        f"- std_floor: `{result['std_floor']}`",
        "",
        "## Output Files",
        "",
        f"- normalizer npz: `{os.path.relpath(result['npz_path'], save_dir)}`",
        f"- combined csv: `{os.path.relpath(result['csv_path'], save_dir)}`",
        f"- combined json: `{os.path.relpath(result['json_path'], save_dir)}`",
        "",
        "## Largest Raw Std",
        "",
        "| dim | name | raw_std | p95-p05 | min | max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for dim in top_std:
        name = _dim_name(int(dim), obs_dim)
        lines.append(
            f"| {int(dim)} | {name} | {stats['raw_std'][dim]:.6g} | "
            f"{(stats['p95'][dim] - stats['p05'][dim]):.6g} | {stats['min'][dim]:.6g} | {stats['max'][dim]:.6g} |"
        )
    lines += [
        "",
        "## Largest P95-P05 Range",
        "",
        "| dim | name | raw_std | p95-p05 | min | max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for dim in top_range:
        name = _dim_name(int(dim), obs_dim)
        lines.append(
            f"| {int(dim)} | {name} | {stats['raw_std'][dim]:.6g} | "
            f"{(stats['p95'][dim] - stats['p05'][dim]):.6g} | {stats['min'][dim]:.6g} | {stats['max'][dim]:.6g} |"
        )
    lines += [
        "",
        "## Std Floor",
        "",
        f"- floored_dims: `{len(floored)}/{obs_dim}`",
        f"- floored_dim_ids: `{', '.join(str(int(dim)) for dim in floored.tolist())}`",
        "",
        "## Plots",
        "",
    ]
    for name, plot_path in result.get("plots", {}).items():
        if name.endswith("error"):
            continue
        lines.append(f"- {name}: `{os.path.relpath(plot_path, save_dir)}`")
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")


def run_collection(args):
    import torch
    import torch.nn as nn

    from tools import eval_comprehensive as base_eval
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

    _load_training_deps()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False

    checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
        checkpoint_step = _checkpoint_step(checkpoint_path)
    else:
        checkpoint_step, checkpoint_path = base_eval._infer_latest_full_checkpoint(checkpoint_dir)
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

    from fdpi_reachability_dreamer.train import build_env

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
    for module in (world_model, agent, gp_critic, gd_critic, dual_policy):
        module.eval()
    modules = (world_model, agent, gp_critic, gd_critic, dual_policy)

    policies = [policy.strip() for policy in str(args.policies).split(",") if policy.strip()]
    if args.smoke:
        policies = policies[:1]

    save_root = os.path.abspath(os.path.expanduser(args.save_dir))
    run_name = str(args.run_name or "").strip()
    if not run_name:
        run_name = f"dfd_v5_obs_normalizer_{checkpoint_step or 'unknown'}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

    arrays = []
    policy_summaries = {}
    policy_counts = {}
    policy_stats = {}
    policy_flat_paths = {}
    try:
        for idx, policy in enumerate(policies):
            rollout_env = first_env if idx == 0 else None
            rollout = base_eval.collect_rollout(args, conf.clone(), modules, policy, idx, vec_env=rollout_env)
            if idx == 0:
                first_env = None
            flat = _flatten_obs_arrays(rollout, include_next_obs=bool(args.include_next_obs))
            arrays.append(flat)
            policy_summaries[policy] = rollout.summary
            policy_counts[policy] = int(flat.shape[0])
            policy_stats[policy] = _stats_from_array(flat, args.std_floor)
            flat_path = os.path.join(save_dir, f"{policy}_flat_obs.npy")
            np.save(flat_path, flat.astype(np.float32, copy=False))
            policy_flat_paths[policy] = flat_path
            _write_csv(os.path.join(save_dir, f"{policy}_obs_normalizer_stats.csv"), policy_stats[policy], obs_dim, args.std_floor)
    finally:
        if first_env is not None:
            first_env.close()

    if not arrays:
        raise RuntimeError("No rollout observations collected.")
    combined = np.concatenate(arrays, axis=0)
    stats = _stats_from_array(combined, args.std_floor)

    npz_path = os.path.join(save_dir, "obs_normalizer.npz")
    np.savez(
        npz_path,
        mean=stats["mean"],
        std=stats["std"],
        raw_std=stats["raw_std"],
        min=stats["min"],
        max=stats["max"],
        p01=stats["p01"],
        p05=stats["p05"],
        p50=stats["p50"],
        p95=stats["p95"],
        p99=stats["p99"],
        obs_dim=np.asarray([obs_dim], dtype=np.int64),
        sample_count=np.asarray([combined.shape[0]], dtype=np.int64),
        std_floor=np.asarray([float(args.std_floor)], dtype=np.float32),
        checkpoint_step=np.asarray([int(checkpoint_step or -1)], dtype=np.int64),
    )
    csv_path = os.path.join(save_dir, "obs_normalizer_stats.csv")
    _write_csv(csv_path, stats, obs_dim, args.std_floor)

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
        "eval_episodes": None if args.eval_episodes is None else int(args.eval_episodes),
        "policies": policies,
        "include_next_obs": bool(args.include_next_obs),
        "std_floor": float(args.std_floor),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "sample_count": int(combined.shape[0]),
        "policy_sample_counts": policy_counts,
        "policy_flat_obs_paths": policy_flat_paths,
        "policy_rollout_summaries": policy_summaries,
        "stats": stats,
        "policy_stats": policy_stats,
        "npz_path": npz_path,
        "csv_path": csv_path,
        "json_path": os.path.join(save_dir, "obs_normalizer_stats.json"),
    }
    result["plots"] = _plot_stats(save_dir, stats, obs_dim, args.std_floor)
    _write_json(result["json_path"], result)
    md_path = os.path.join(save_dir, "dfd_v5_obs_normalizer.md")
    result["md_path"] = md_path
    _write_report(md_path, result, stats, obs_dim, save_dir)

    print("\nDFD v5 obs normalizer collection")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config: {config_path}")
    print(f"  save_dir: {save_dir}")
    print(f"  policies: {', '.join(policies)}")
    print(f"  samples: {combined.shape[0]}")
    print(f"  npz: {npz_path}")
    print(f"  csv: {csv_path}")
    print(f"  md: {md_path}")
    return result


def main():
    warnings.filterwarnings("ignore")
    from tools import eval_comprehensive as base_eval

    default_ckpt_dir = os.path.join(
        PROJECT_ROOT,
        "ckpt",
        "dfd-v5-task-first-from-750k",
        "20260603_121444",
    )
    parser = argparse.ArgumentParser(description="Collect fixed observation-normalization parameters for DFD v5.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1")
    parser.add_argument("--device", "-device", type=str, default="cuda:0")
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=65536)
    parser.add_argument("--eval_episodes", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--policies", type=str, default="main,dual,random")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--std_floor", type=float, default=1.0)
    parser.add_argument("--include_next_obs", action="store_true", default=True)
    parser.add_argument("--obs_only", action="store_false", dest="include_next_obs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_obs_normalizer"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.greedy = not args.stochastic

    base_eval._launch_isaac(headless=not args.render)
    try:
        run_collection(args)
    finally:
        if base_eval.simulation_app is not None:
            base_eval.simulation_app.close()


if __name__ == "__main__":
    main()
