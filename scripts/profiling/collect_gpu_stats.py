from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from datetime import datetime


GPU_QUERY = (
    "timestamp,name,utilization.gpu,memory.used,memory.total,"
    "utilization.memory,power.draw,temperature.gpu"
)

CSV_FIELDS = (
    "timestamp",
    "gpu_name",
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_memory_utilization_percent",
    "gpu_power_draw_w",
    "gpu_temperature_c",
)

_STOP = False


def _handle_stop(signum, frame):
    del signum, frame
    global _STOP
    _STOP = True


def _parse_float(value, default=0.0):
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def query_gpu_stats():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for raw_line in output.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 8:
            continue
        timestamp, name, util, mem_used, mem_total, mem_util, power, temp = parts[:8]
        rows.append(
            {
                "timestamp": timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
                "gpu_name": name,
                "gpu_utilization_percent": _parse_float(util),
                "gpu_memory_used_mib": _parse_float(mem_used),
                "gpu_memory_total_mib": _parse_float(mem_total),
                "gpu_memory_utilization_percent": _parse_float(mem_util),
                "gpu_power_draw_w": _parse_float(power),
                "gpu_temperature_c": _parse_float(temp),
            }
        )
    return rows


def collect(output_path, interval_seconds=1.0, duration_seconds=0.0):
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start = time.monotonic()
    wrote_header = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        if not wrote_header:
            writer.writeheader()
            fout.flush()
        while True:
            if _STOP:
                break
            try:
                rows = query_gpu_stats()
            except Exception as exc:
                rows = [
                    {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "gpu_name": f"nvidia-smi-error: {exc}",
                        "gpu_utilization_percent": 0.0,
                        "gpu_memory_used_mib": 0.0,
                        "gpu_memory_total_mib": 0.0,
                        "gpu_memory_utilization_percent": 0.0,
                        "gpu_power_draw_w": 0.0,
                        "gpu_temperature_c": 0.0,
                    }
                ]
            for row in rows:
                writer.writerow(row)
            fout.flush()
            if duration_seconds and time.monotonic() - start >= float(duration_seconds):
                break
            sleep_until = time.monotonic() + max(float(interval_seconds), 0.1)
            while not _STOP and time.monotonic() < sleep_until:
                time.sleep(min(0.1, max(sleep_until - time.monotonic(), 0.0)))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect GPU utilization with nvidia-smi.")
    parser.add_argument("--output", required=True, help="CSV path to write GPU stats.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 means run until killed.")
    return parser.parse_args()


def main():
    args = parse_args()
    collect(args.output, interval_seconds=args.interval, duration_seconds=args.duration)


if __name__ == "__main__":
    main()
