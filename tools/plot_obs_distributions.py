from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

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


POLICY_COLORS = {
    "main": "#2d6a9f",
    "dual": "#b45f06",
    "random": "#2f7d32",
    "combined": "#4b4f56",
}


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "obs"


def _dim_name(dim: int, obs_dim: int) -> str:
    if obs_dim == len(OBS_DIM_NAMES):
        return OBS_DIM_NAMES[dim]
    return f"obs_{dim}"


def _dim_group(name: str) -> str:
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


def _find_flat_obs(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    candidates = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith("_flat_obs.npy")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_flat_obs.npy found in {path}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple *_flat_obs.npy files in {path}; pass files explicitly.")
    return candidates[0]


def _policy_name(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith("_flat_obs.npy"):
        return name[: -len("_flat_obs.npy")]
    return os.path.splitext(name)[0]


def _load_arrays(inputs: list[str]) -> dict[str, np.ndarray]:
    arrays = {}
    for item in inputs:
        path = _find_flat_obs(item)
        policy = _policy_name(path)
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D flat obs array in {path}, got {arr.shape}")
        arrays[policy] = arr
    if not arrays:
        raise ValueError("No arrays loaded.")
    obs_dim = {arr.shape[1] for arr in arrays.values()}
    if len(obs_dim) != 1:
        raise ValueError(f"Mismatched obs dims: {sorted(obs_dim)}")
    return arrays


def _sample_for_plot(arr: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or arr.shape[0] <= max_samples:
        return arr
    idx = rng.choice(arr.shape[0], size=max_samples, replace=False)
    return arr[idx]


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return values[np.isfinite(values)]


def _xlim(values: np.ndarray, percentile_low: float, percentile_high: float) -> tuple[float, float]:
    values = _finite(values)
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(values, percentile_low))
    hi = float(np.nanpercentile(values, percentile_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmean(values)) if values.size else 0.0
        span = max(abs(center) * 0.05, 1.0e-3)
        return center - span, center + span
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _write_stats_csv(path: str, arrays: dict[str, np.ndarray], combined: np.ndarray) -> None:
    obs_dim = combined.shape[1]
    fields = [
        "policy",
        "obs_dim",
        "name",
        "group",
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
        "zero_rate",
        "clip_neg50_rate",
        "clip_pos50_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for policy, arr in {"combined": combined, **arrays}.items():
            for dim in range(obs_dim):
                values = _finite(arr[:, dim])
                name = _dim_name(dim, obs_dim)
                if values.size == 0:
                    row = {key: "" for key in fields}
                    row.update({"policy": policy, "obs_dim": dim, "name": name, "group": _dim_group(name), "count": 0})
                    writer.writerow(row)
                    continue
                row = {
                    "policy": policy,
                    "obs_dim": dim,
                    "name": name,
                    "group": _dim_group(name),
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p01": float(np.percentile(values, 1)),
                    "p05": float(np.percentile(values, 5)),
                    "p25": float(np.percentile(values, 25)),
                    "p50": float(np.percentile(values, 50)),
                    "p75": float(np.percentile(values, 75)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(values.max()),
                    "zero_rate": float(np.mean(np.isclose(values, 0.0, atol=1.0e-8))),
                    "clip_neg50_rate": float(np.mean(values <= -49.999)),
                    "clip_pos50_rate": float(np.mean(values >= 49.999)),
                }
                writer.writerow(row)


def _plot_overview_combined(save_dir, combined, obs_dim, bins, percentile_low, percentile_high):
    import matplotlib.pyplot as plt

    cols = 4
    rows = int(np.ceil(obs_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.0 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for dim in range(obs_dim):
        ax = axes[dim]
        values = _finite(combined[:, dim])
        lo, hi = _xlim(values, percentile_low, percentile_high)
        ax.hist(values, bins=bins, range=(lo, hi), density=True, color="#4b4f56", alpha=0.78)
        ax.axvline(float(np.mean(values)), color="#c43c35", linewidth=1.0, alpha=0.9)
        ax.set_title(f"{dim}: {_dim_name(dim, obs_dim)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    for ax in axes[obs_dim:]:
        ax.set_axis_off()
    path = os.path.join(save_dir, "obs_distribution_grid_combined.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_overview_by_policy(save_dir, arrays, combined, obs_dim, bins, percentile_low, percentile_high, max_samples, rng):
    import matplotlib.pyplot as plt

    sampled = {policy: _sample_for_plot(arr, max_samples, rng) for policy, arr in arrays.items()}
    cols = 4
    rows = int(np.ceil(obs_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.0 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for dim in range(obs_dim):
        ax = axes[dim]
        lo, hi = _xlim(combined[:, dim], percentile_low, percentile_high)
        for policy, arr in sampled.items():
            values = _finite(arr[:, dim])
            ax.hist(
                values,
                bins=bins,
                range=(lo, hi),
                density=True,
                histtype="step",
                linewidth=1.1,
                color=POLICY_COLORS.get(policy, None),
                label=policy,
            )
        ax.set_title(f"{dim}: {_dim_name(dim, obs_dim)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
        if dim == 0:
            ax.legend(fontsize=7)
    for ax in axes[obs_dim:]:
        ax.set_axis_off()
    path = os.path.join(save_dir, "obs_distribution_grid_by_policy.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_group_grids(save_dir, arrays, combined, obs_dim, bins, percentile_low, percentile_high, max_samples, rng):
    import matplotlib.pyplot as plt

    sampled = {policy: _sample_for_plot(arr, max_samples, rng) for policy, arr in arrays.items()}
    groups = defaultdict(list)
    for dim in range(obs_dim):
        groups[_dim_group(_dim_name(dim, obs_dim))].append(dim)

    paths = {}
    group_dir = os.path.join(save_dir, "groups")
    os.makedirs(group_dir, exist_ok=True)
    for group, dims in groups.items():
        cols = min(3, len(dims))
        rows = int(np.ceil(len(dims) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, dim in zip(axes, dims):
            lo, hi = _xlim(combined[:, dim], percentile_low, percentile_high)
            for policy, arr in sampled.items():
                values = _finite(arr[:, dim])
                ax.hist(
                    values,
                    bins=bins,
                    range=(lo, hi),
                    density=True,
                    histtype="step",
                    linewidth=1.2,
                    color=POLICY_COLORS.get(policy, None),
                    label=policy,
                )
            ax.set_title(f"{dim}: {_dim_name(dim, obs_dim)}", fontsize=9)
            ax.grid(alpha=0.2)
        for ax in axes[len(dims):]:
            ax.set_axis_off()
        axes[0].legend(fontsize=8)
        fig.suptitle(group, fontsize=13)
        path = os.path.join(group_dir, f"obs_distribution_group_{_safe_name(group)}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths[group] = path
    return paths


def _plot_percentile_overview(save_dir, combined, obs_dim):
    import matplotlib.pyplot as plt

    dims = np.arange(obs_dim)
    p01 = np.percentile(combined, 1, axis=0)
    p05 = np.percentile(combined, 5, axis=0)
    p50 = np.percentile(combined, 50, axis=0)
    p95 = np.percentile(combined, 95, axis=0)
    p99 = np.percentile(combined, 99, axis=0)

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.vlines(dims, p01, p99, color="#8aa0ad", linewidth=2.0, alpha=0.75, label="p01-p99")
    ax.vlines(dims, p05, p95, color="#2d6a9f", linewidth=4.0, alpha=0.85, label="p05-p95")
    ax.scatter(dims, p50, color="#c43c35", s=14, label="median", zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Observation percentile ranges by dimension")
    ax.set_xlabel("obs dim")
    ax.set_ylabel("value")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    path = os.path.join(save_dir, "obs_dim_percentile_ranges.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_clip_rates(save_dir, combined, obs_dim):
    import matplotlib.pyplot as plt

    dims = np.arange(obs_dim)
    neg = np.mean(combined <= -49.999, axis=0)
    pos = np.mean(combined >= 49.999, axis=0)
    zero = np.mean(np.isclose(combined, 0.0, atol=1.0e-8), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(dims - 0.18, neg, width=0.36, label="<= -50", color="#7b506f")
    axes[0].bar(dims + 0.18, pos, width=0.36, label=">= 50", color="#b36b31")
    axes[0].set_title("Distance clip saturation rate")
    axes[0].set_ylabel("rate")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.22, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].bar(dims, zero, width=0.75, color="#4b4f56")
    axes[1].set_title("Exact-zero rate")
    axes[1].set_xlabel("obs dim")
    axes[1].set_ylabel("rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.22, axis="y")
    path = os.path.join(save_dir, "obs_dim_clip_zero_rates.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_individual_dims(save_dir, arrays, combined, obs_dim, bins, percentile_low, percentile_high, max_samples, rng):
    import matplotlib.pyplot as plt

    per_dim_dir = os.path.join(save_dir, "per_dim")
    os.makedirs(per_dim_dir, exist_ok=True)
    sampled = {policy: _sample_for_plot(arr, max_samples, rng) for policy, arr in arrays.items()}
    paths = []
    for dim in range(obs_dim):
        name = _dim_name(dim, obs_dim)
        values_all = _finite(combined[:, dim])
        lo, hi = _xlim(values_all, percentile_low, percentile_high)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
        ax = axes[0]
        ax.hist(
            values_all,
            bins=bins,
            range=(lo, hi),
            density=True,
            color=POLICY_COLORS["combined"],
            alpha=0.18,
            label="combined",
        )
        for policy, arr in sampled.items():
            values = _finite(arr[:, dim])
            ax.hist(
                values,
                bins=bins,
                range=(lo, hi),
                density=True,
                histtype="step",
                linewidth=1.5,
                color=POLICY_COLORS.get(policy, None),
                label=policy,
            )
        mean = float(values_all.mean())
        median = float(np.percentile(values_all, 50))
        ax.axvline(mean, color="#c43c35", linewidth=1.2, label="combined mean")
        ax.axvline(median, color="#202124", linewidth=1.0, linestyle="--", label="combined median")
        ax.set_title(f"obs {dim}: {name}")
        ax.set_ylabel("density")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)

        box_values = [_finite(combined[:, dim])] + [_finite(arr[:, dim]) for arr in sampled.values()]
        labels = ["combined"] + list(sampled.keys())
        axes[1].boxplot(box_values, vert=False, labels=labels, showfliers=False)
        axes[1].set_xlim(lo, hi)
        axes[1].grid(alpha=0.22, axis="x")
        axes[1].set_xlabel("value")
        path = os.path.join(per_dim_dir, f"obs_{dim:02d}_{_safe_name(name)}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_distributions(args):
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(int(args.seed))
    arrays = _load_arrays(args.inputs)
    combined = np.concatenate(list(arrays.values()), axis=0)
    obs_dim = combined.shape[1]
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    stats_csv = os.path.join(save_dir, "obs_distribution_stats.csv")
    _write_stats_csv(stats_csv, arrays, combined)

    plot_paths = {}
    plot_paths["combined_grid"] = _plot_overview_combined(
        save_dir,
        combined,
        obs_dim,
        args.bins,
        args.percentile_low,
        args.percentile_high,
    )
    plot_paths["policy_grid"] = _plot_overview_by_policy(
        save_dir,
        arrays,
        combined,
        obs_dim,
        args.bins,
        args.percentile_low,
        args.percentile_high,
        args.max_plot_samples,
        rng,
    )
    plot_paths["percentile_ranges"] = _plot_percentile_overview(save_dir, combined, obs_dim)
    plot_paths["clip_zero_rates"] = _plot_clip_rates(save_dir, combined, obs_dim)
    group_paths = _plot_group_grids(
        save_dir,
        arrays,
        combined,
        obs_dim,
        args.bins,
        args.percentile_low,
        args.percentile_high,
        args.max_plot_samples,
        rng,
    )
    per_dim_paths = []
    if not args.skip_per_dim:
        per_dim_paths = _plot_individual_dims(
            save_dir,
            arrays,
            combined,
            obs_dim,
            args.bins,
            args.percentile_low,
            args.percentile_high,
            args.max_plot_samples,
            rng,
        )

    md_path = os.path.join(save_dir, "obs_distribution_report.md")
    top_std = np.argsort(-np.std(combined, axis=0))[:12]
    with open(md_path, "w", encoding="utf-8") as fout:
        fout.write("# DFD v5 Observation Distributions\n\n")
        fout.write(f"- sample_count: `{combined.shape[0]}`\n")
        fout.write(f"- obs_dim: `{obs_dim}`\n")
        fout.write(f"- policies: `{', '.join(arrays.keys())}`\n")
        fout.write(f"- stats_csv: `{os.path.relpath(stats_csv, save_dir)}`\n\n")
        fout.write("## Plots\n\n")
        for name, path in plot_paths.items():
            fout.write(f"- {name}: `{os.path.relpath(path, save_dir)}`\n")
        fout.write(f"- group_plots_dir: `groups/`\n")
        if per_dim_paths:
            fout.write(f"- per_dim_plots_dir: `per_dim/` ({len(per_dim_paths)} files)\n")
        fout.write("\n## Largest Std Dimensions\n\n")
        fout.write("| dim | name | std | p05 | p50 | p95 | clip -50 | clip +50 |\n")
        fout.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for dim in top_std:
            values = combined[:, dim]
            fout.write(
                f"| {int(dim)} | {_dim_name(int(dim), obs_dim)} | {float(np.std(values)):.6g} | "
                f"{float(np.percentile(values, 5)):.6g} | {float(np.percentile(values, 50)):.6g} | "
                f"{float(np.percentile(values, 95)):.6g} | "
                f"{float(np.mean(values <= -49.999)):.6g} | {float(np.mean(values >= 49.999)):.6g} |\n"
            )

    print("DFD v5 obs distribution plots")
    print(f"  save_dir: {save_dir}")
    print(f"  samples: {combined.shape[0]}")
    print(f"  stats_csv: {stats_csv}")
    print(f"  report: {md_path}")
    print(f"  combined_grid: {plot_paths['combined_grid']}")
    print(f"  policy_grid: {plot_paths['policy_grid']}")
    print(f"  per_dim_count: {len(per_dim_paths)}")


def main():
    default_root = os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_obs_normalizer_policy")
    default_inputs = [
        os.path.join(default_root, "main_9749952_64env_256ep"),
        os.path.join(default_root, "dual_9749952_64env_256ep"),
        os.path.join(default_root, "random_9749952_64env_256ep"),
    ]
    default_save = os.path.join(
        PROJECT_ROOT,
        "eval_results",
        "dfd_v5_obs_distributions",
        "dfd_v5_9749952_main_dual_random_64env_256ep",
    )
    parser = argparse.ArgumentParser(description="Plot per-dimension DFD v5 observation distributions.")
    parser.add_argument("inputs", nargs="*", default=default_inputs)
    parser.add_argument("--save_dir", type=str, default=default_save)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--percentile_low", type=float, default=0.5)
    parser.add_argument("--percentile_high", type=float, default=99.5)
    parser.add_argument("--max_plot_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_per_dim", action="store_true")
    args = parser.parse_args()
    plot_distributions(args)


if __name__ == "__main__":
    main()
