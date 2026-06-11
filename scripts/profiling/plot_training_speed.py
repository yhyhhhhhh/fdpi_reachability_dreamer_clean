from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime


def _load_json(path, default):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def _float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _parse_time(value, fallback_idx):
    text = str(value or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    return float(fallback_idx)


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save_no_data_plot(path, title, message):
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_throughput(metrics, timeline, image_dir):
    path = os.path.join(image_dir, "图1_训练吞吐量曲线.png")
    if not timeline:
        _save_no_data_plot(path, "训练吞吐量曲线", "缺少吞吐量时间序列")
        return path
    plt = _ensure_matplotlib()
    x = [int(row.get("env_steps", idx)) for idx, row in enumerate(timeline)]
    env_sps = [_float(row, "env_steps_per_second") for row in timeline]
    update_sps = [_float(row, "train_updates_per_second") for row in timeline]
    sample_sps = [_float(row, "samples_per_second") for row in timeline]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, env_sps, marker="o", label="env steps/s")
    ax.plot(x, update_sps, marker="s", label="train updates/s")
    ax.plot(x, sample_sps, marker="^", label="samples/s")
    ax.set_title("训练吞吐量曲线")
    ax.set_xlabel("环境步数")
    ax.set_ylabel("吞吐量")
    ax.grid(True, alpha=0.25)
    ax.legend()
    summary = metrics.get("summary", {})
    if summary:
        text = f"平均 env steps/s: {summary.get('env_steps_per_second', 0.0):.2f}"
        ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_gpu_util(gpu_rows, image_dir):
    path = os.path.join(image_dir, "图2_GPU利用率曲线.png")
    if not gpu_rows:
        _save_no_data_plot(path, "GPU利用率曲线", "缺少 GPU 采样数据")
        return path
    plt = _ensure_matplotlib()
    t0 = _parse_time(gpu_rows[0].get("timestamp"), 0)
    x = [_parse_time(row.get("timestamp"), idx) - t0 for idx, row in enumerate(gpu_rows)]
    util = [_float(row, "gpu_utilization_percent") for row in gpu_rows]
    mem_util = [_float(row, "gpu_memory_utilization_percent") for row in gpu_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, util, label="GPU util %")
    ax.plot(x, mem_util, label="Memory util %")
    ax.set_title("GPU利用率曲线")
    ax.set_xlabel("运行时间（秒）")
    ax.set_ylabel("利用率（%）")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_gpu_memory(gpu_rows, image_dir):
    path = os.path.join(image_dir, "图3_显存占用曲线.png")
    if not gpu_rows:
        _save_no_data_plot(path, "显存占用曲线", "缺少 GPU 采样数据")
        return path
    plt = _ensure_matplotlib()
    t0 = _parse_time(gpu_rows[0].get("timestamp"), 0)
    x = [_parse_time(row.get("timestamp"), idx) - t0 for idx, row in enumerate(gpu_rows)]
    used = [_float(row, "gpu_memory_used_mib") for row in gpu_rows]
    total = max([_float(row, "gpu_memory_total_mib") for row in gpu_rows] or [0.0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, used, label="Memory used MiB")
    if total > 0:
        ax.axhline(total, linestyle="--", color="gray", alpha=0.7, label=f"Total {total:.0f} MiB")
    ax.set_title("显存占用曲线")
    ax.set_xlabel("运行时间（秒）")
    ax.set_ylabel("显存（MiB）")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_phase_share(metrics, image_dir):
    path = os.path.join(image_dir, "图4_阶段耗时占比.png")
    phase_seconds = metrics.get("phase_seconds", {})
    labels = [
        ("policy_inference_time", "策略推理"),
        ("env_step_time", "环境 step"),
        ("replay_insert_time", "replay 写入"),
        ("sample_batch_time", "batch 采样"),
        ("world_model_update_time", "world model 更新"),
        ("actor_critic_update_time", "actor/critic 更新"),
        ("logging_time", "日志"),
        ("checkpoint_time", "checkpoint"),
    ]
    values = [(label, float(phase_seconds.get(key, 0.0))) for key, label in labels]
    values = [(label, value) for label, value in values if value > 0]
    if not values:
        _save_no_data_plot(path, "阶段耗时占比", "缺少阶段耗时数据")
        return path
    plt = _ensure_matplotlib()
    names, seconds = zip(*values)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6", "#9D755D"]
    ax.pie(seconds, labels=names, autopct="%1.1f%%", startangle=90, colors=colors[: len(seconds)])
    ax.set_title("阶段耗时占比")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_experiment(experiment_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    image_dir = os.path.join(experiment_dir, "图片")
    os.makedirs(image_dir, exist_ok=True)
    metrics = _load_json(os.path.join(experiment_dir, "指标结果.json"), {})
    timeline = _load_csv(os.path.join(experiment_dir, "日志", "training_speed_timeline.csv"))
    gpu_rows = _load_csv(os.path.join(experiment_dir, "日志", "gpu_stats.csv"))
    generated = [
        plot_throughput(metrics, timeline, image_dir),
        plot_gpu_util(gpu_rows, image_dir),
        plot_gpu_memory(gpu_rows, image_dir),
        plot_phase_share(metrics, image_dir),
    ]
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training-speed and GPU profiling results.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory containing 指标结果.json and 日志/.")
    return parser.parse_args()


def main():
    args = parse_args()
    generated = plot_experiment(args.experiment_dir)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
