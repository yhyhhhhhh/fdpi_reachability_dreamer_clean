from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_COLLECT_PATH = os.path.join(os.path.dirname(__file__), "collect_obs_normalizer.py")
_SPEC = importlib.util.spec_from_file_location("_dfd_v5_obs_normalizer_collect_helpers", _COLLECT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load helper module from {_COLLECT_PATH}")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

_plot_stats = _HELPERS._plot_stats
_stats_from_array = _HELPERS._stats_from_array
_write_csv = _HELPERS._write_csv
_write_json = _HELPERS._write_json
_write_report = _HELPERS._write_report


def _find_flat_obs_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing input path: {path}")
    candidates = []
    for name in os.listdir(path):
        if name.endswith("_flat_obs.npy"):
            candidates.append(os.path.join(path, name))
    if not candidates:
        raise FileNotFoundError(f"No *_flat_obs.npy found in {path}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple *_flat_obs.npy files found in {path}; pass files explicitly.")
    return candidates[0]


def _policy_from_path(path):
    name = os.path.basename(path)
    if name.endswith("_flat_obs.npy"):
        return name[: -len("_flat_obs.npy")]
    return os.path.splitext(name)[0]


def _load_metadata(flat_path):
    directory = os.path.dirname(flat_path)
    meta_path = os.path.join(directory, "obs_normalizer_stats.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def merge_normalizers(args):
    flat_paths = [_find_flat_obs_path(path) for path in args.inputs]
    arrays = []
    policy_counts = {}
    policy_stats = {}
    policy_flat_paths = {}
    metadata = [_load_metadata(path) for path in flat_paths]

    for path in flat_paths:
        policy = _policy_from_path(path)
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D flat obs array in {path}, got shape {arr.shape}")
        arrays.append(arr)
        policy_counts[policy] = int(arr.shape[0])
        policy_stats[policy] = _stats_from_array(arr, args.std_floor)
        policy_flat_paths[policy] = path

    combined = np.concatenate(arrays, axis=0)
    obs_dim = int(combined.shape[-1])
    stats = _stats_from_array(combined, args.std_floor)

    save_root = os.path.abspath(os.path.expanduser(args.save_dir))
    run_name = args.run_name or f"dfd_v5_obs_normalizer_merged_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

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
    )
    csv_path = os.path.join(save_dir, "obs_normalizer_stats.csv")
    _write_csv(csv_path, stats, obs_dim, args.std_floor)
    for policy, item in policy_stats.items():
        _write_csv(os.path.join(save_dir, f"{policy}_obs_normalizer_stats.csv"), item, obs_dim, args.std_floor)

    first_meta = next((item for item in metadata if item), {})
    policies = list(policy_counts.keys())
    result = {
        "checkpoint_dir": first_meta.get("checkpoint_dir"),
        "checkpoint_path": first_meta.get("checkpoint_path"),
        "checkpoint_step": first_meta.get("checkpoint_step"),
        "config_path": first_meta.get("config_path"),
        "env_name": first_meta.get("env_name"),
        "device": first_meta.get("device"),
        "seed": first_meta.get("seed"),
        "num_envs": first_meta.get("num_envs"),
        "eval_steps": first_meta.get("eval_steps"),
        "eval_episodes": first_meta.get("eval_episodes"),
        "policies": policies,
        "include_next_obs": first_meta.get("include_next_obs"),
        "std_floor": float(args.std_floor),
        "obs_dim": obs_dim,
        "action_dim": first_meta.get("action_dim"),
        "sample_count": int(combined.shape[0]),
        "policy_sample_counts": policy_counts,
        "policy_flat_obs_paths": policy_flat_paths,
        "policy_rollout_summaries": {
            policy: meta.get("policy_rollout_summaries", {}).get(policy)
            for policy, meta in zip(policies, metadata)
            if meta.get("policy_rollout_summaries", {}).get(policy) is not None
        },
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

    print("Merged DFD v5 obs normalizer")
    print(f"  save_dir: {save_dir}")
    print(f"  policies: {', '.join(policies)}")
    print(f"  samples: {combined.shape[0]}")
    print(f"  npz: {npz_path}")
    print(f"  csv: {csv_path}")
    print(f"  md: {md_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge per-policy DFD v5 obs normalizer samples.")
    parser.add_argument("inputs", nargs="+", help="Directories containing *_flat_obs.npy or flat obs .npy files.")
    parser.add_argument("--std_floor", type=float, default=1.0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_obs_normalizer_merged"),
    )
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    merge_normalizers(args)


if __name__ == "__main__":
    main()
