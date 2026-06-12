from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .cost_utils import SOURCE_DUAL, SOURCE_MAIN, cfg_get


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def source_cost_weight(
    cost: torch.Tensor,
    source: torch.Tensor | None,
    *,
    high_cost_weight: float = 1.0,
    dual_source_weight: float = 1.0,
    boundary_weight: float = 1.0,
    high_cost_threshold: float = 0.1,
    boundary_low: float = 0.05,
    boundary_high: float = 0.4,
    normalize: bool = False,
) -> torch.Tensor:
    cost = torch.as_tensor(cost, dtype=torch.float32)
    weight = torch.ones_like(cost)
    if source is not None:
        source = torch.as_tensor(source, device=cost.device).to(torch.int64)
        weight = torch.where(source == SOURCE_DUAL, weight * float(dual_source_weight), weight)
    high = cost > float(high_cost_threshold)
    boundary = (cost >= float(boundary_low)) & (cost <= float(boundary_high))
    weight = torch.where(high, weight * float(high_cost_weight), weight)
    weight = torch.where(boundary, weight * float(boundary_weight), weight)
    if normalize:
        weight = weight / weight.mean().detach().clamp_min(1.0e-6)
    return weight


def batch_composition(
    batch: dict,
    *,
    high_cost_threshold: float = 0.1,
    boundary_low: float = 0.05,
    boundary_high: float = 0.4,
) -> dict[str, float]:
    cost = torch.as_tensor(batch.get("continuous_cost", batch.get("cost")), dtype=torch.float32).reshape(-1)
    source = torch.as_tensor(batch.get("source", torch.zeros_like(cost)), device=cost.device).to(torch.int64).reshape(-1)
    if cost.numel() == 0:
        return {
            "source_dual_ratio": 0.0,
            "high_cost_ratio": 0.0,
            "boundary_ratio": 0.0,
        }
    high = cost > float(high_cost_threshold)
    boundary = (cost >= float(boundary_low)) & (cost <= float(boundary_high))
    return {
        "source_dual_ratio": float((source == SOURCE_DUAL).float().mean().item()),
        "high_cost_ratio": float(high.float().mean().item()),
        "boundary_ratio": float(boundary.float().mean().item()),
    }


@dataclass
class FDPIRegimeStats:
    fea_ratio: float = 0.0
    cri_ratio: float = 0.0
    inf_ratio: float = 0.0
    main_real_cost_rate: float = 0.0
    count: int = 0


class SourceCostStatsWindow:
    def __init__(self, maxlen: int):
        self.maxlen = max(int(maxlen), 1)
        self.rows: deque[dict[str, torch.Tensor | int]] = deque()
        self.total = 0
        self.sums: dict[str, torch.Tensor] = {}

    def append(
        self,
        *,
        source: torch.Tensor,
        continuous_cost: torch.Tensor,
        binary_cost: torch.Tensor,
        extreme_cost: torch.Tensor | None = None,
    ) -> None:
        source = torch.as_tensor(source).detach().reshape(-1).to(torch.int64)
        continuous_cost = torch.as_tensor(continuous_cost, device=source.device).detach().reshape(-1)
        binary_cost = torch.as_tensor(binary_cost, device=source.device).detach().reshape(-1)
        if extreme_cost is None:
            extreme_cost = torch.zeros_like(binary_cost)
        else:
            extreme_cost = torch.as_tensor(extreme_cost, device=source.device).detach().reshape(-1)

        main_mask = source == SOURCE_MAIN
        dual_mask = source == SOURCE_DUAL
        main_float = main_mask.to(dtype=torch.float32)
        dual_float = dual_mask.to(dtype=torch.float32)
        binary_positive = (binary_cost > 0.0).to(dtype=torch.float32)
        extreme_positive = (extreme_cost > 0.0).to(dtype=torch.float32)
        row = {
            "total": int(source.numel()),
            "main_total": main_float.sum(),
            "dual_total": dual_float.sum(),
            "main_binary_cost": (binary_positive * main_float).sum(),
            "dual_binary_cost": (binary_positive * dual_float).sum(),
            "main_continuous_cost": (continuous_cost.float() * main_float).sum(),
            "dual_continuous_cost": (continuous_cost.float() * dual_float).sum(),
            "main_extreme_cost": (extreme_positive * main_float).sum(),
            "dual_extreme_cost": (extreme_positive * dual_float).sum(),
        }
        self.rows.append(row)
        self._add(row)
        while self.total > self.maxlen and len(self.rows) > 1:
            self._subtract(self.rows.popleft())

    def _add(self, row: dict[str, torch.Tensor | int]) -> None:
        self.total += int(row["total"])
        for key, value in row.items():
            if key == "total":
                continue
            value = torch.as_tensor(value).detach()
            if key not in self.sums:
                self.sums[key] = value.clone()
            else:
                self.sums[key] = self.sums[key] + value.to(device=self.sums[key].device)

    def _subtract(self, row: dict[str, torch.Tensor | int]) -> None:
        self.total -= int(row["total"])
        for key, value in row.items():
            if key == "total" or key not in self.sums:
                continue
            value = torch.as_tensor(value).detach().to(device=self.sums[key].device)
            self.sums[key] = self.sums[key] - value

    def stats(self) -> dict[str, float]:
        def value(key: str) -> float:
            item = self.sums.get(key)
            if item is None:
                return 0.0
            return float(item.detach().float().item())

        main_total_value = value("main_total")
        dual_total_value = value("dual_total")
        total = max(float(self.total), 1.0)
        main_total = max(main_total_value, 1.0)
        dual_total = max(dual_total_value, 1.0)
        return {
            "window_count": float(self.total),
            "main_count": main_total_value,
            "dual_count": dual_total_value,
            "main_ratio": main_total_value / total,
            "dual_ratio": dual_total_value / total,
            "main_binary_cost_rate": value("main_binary_cost") / main_total,
            "dual_binary_cost_rate": value("dual_binary_cost") / dual_total,
            "main_continuous_cost_mean": value("main_continuous_cost") / main_total,
            "dual_continuous_cost_mean": value("dual_continuous_cost") / dual_total,
            "main_extreme_cost_rate": value("main_extreme_cost") / main_total,
            "dual_extreme_cost_rate": value("dual_extreme_cost") / dual_total,
        }


class FDPIRegimeStatsWindow:
    def __init__(self, maxlen: int):
        self.maxlen = max(int(maxlen), 1)
        self.rows: deque[dict[str, int]] = deque()
        self.total = 0
        self.fea = 0
        self.cri = 0
        self.inf = 0
        self.main_cost = 0

    def append(
        self,
        *,
        g_main: torch.Tensor,
        source: torch.Tensor,
        continuous_cost: torch.Tensor,
        pf: float,
        cg: float,
    ) -> None:
        g_main = torch.as_tensor(g_main).detach().reshape(-1)
        source = torch.as_tensor(source, device=g_main.device).detach().reshape(-1).to(torch.int64)
        continuous_cost = torch.as_tensor(continuous_cost, device=g_main.device).detach().reshape(-1)
        main_mask = source == SOURCE_MAIN
        if not bool(main_mask.any().item()):
            return
        g = g_main[main_mask]
        c = continuous_cost[main_mask]
        fea_mask = g < (float(pf) - float(cg))
        cri_mask = (g >= (float(pf) - float(cg))) & (g < float(pf))
        inf_mask = g >= float(pf)
        row = {
            "total": int(g.numel()),
            "fea": int(fea_mask.sum().item()),
            "cri": int(cri_mask.sum().item()),
            "inf": int(inf_mask.sum().item()),
            "main_cost": int((c > 0.0).sum().item()),
        }
        self.rows.append(row)
        self.total += row["total"]
        self.fea += row["fea"]
        self.cri += row["cri"]
        self.inf += row["inf"]
        self.main_cost += row["main_cost"]
        while self.total > self.maxlen and len(self.rows) > 1:
            old = self.rows.popleft()
            self.total -= old["total"]
            self.fea -= old["fea"]
            self.cri -= old["cri"]
            self.inf -= old["inf"]
            self.main_cost -= old["main_cost"]

    def stats(self) -> FDPIRegimeStats:
        if self.total <= 0:
            return FDPIRegimeStats()
        denom = float(max(self.total, 1))
        return FDPIRegimeStats(
            fea_ratio=float(self.fea) / denom,
            cri_ratio=float(self.cri) / denom,
            inf_ratio=float(self.inf) / denom,
            main_real_cost_rate=float(self.main_cost) / denom,
            count=int(self.total),
        )


def dual_ratio_from_fdpi_stats(
    *,
    step: int,
    cfg,
    stats: FDPIRegimeStats,
    last_dual_kl: float,
) -> tuple[float, dict[str, float]]:
    enabled = bool(cfg_get(cfg, "Enable", True))
    start_step = int(cfg_get(cfg, "StartStep", 100000))
    max_kl = float(cfg_get(cfg, "MaxKLForSampling", 200.0))
    high_main_cost_rate = float(cfg_get(cfg, "HighMainCostRate", 0.20))
    max_ratio_when_main_cost_high = float(cfg_get(cfg, "MaxRatioWhenMainCostHigh", 0.10))

    step_ready = int(step) >= start_step
    kl_healthy = abs(float(last_dual_kl)) <= max_kl
    stats_ready = int(stats.count) > 0
    ratio = 0.0
    if enabled and step_ready and kl_healthy and stats_ready:
        if stats.fea_ratio >= 0.95:
            ratio = float(cfg_get(cfg, "RatioFea95", 0.20))
        elif stats.fea_ratio >= 0.90:
            ratio = float(cfg_get(cfg, "RatioFea90", 0.35))
        elif stats.fea_ratio >= 0.80:
            ratio = float(cfg_get(cfg, "RatioFea80", 0.20))
        elif stats.cri_ratio >= 0.30:
            ratio = float(cfg_get(cfg, "RatioCriticalHigh", 0.15))
        elif stats.inf_ratio >= 0.20:
            ratio = float(cfg_get(cfg, "RatioUnsafeHigh", 0.05))
        else:
            ratio = float(cfg_get(cfg, "RatioDefault", 0.10))
        if stats.main_real_cost_rate > high_main_cost_rate:
            ratio = min(ratio, max_ratio_when_main_cost_high)
    info = {
        "enabled": float(enabled),
        "step_ready": float(step_ready),
        "kl_healthy": float(kl_healthy),
        "stats_ready": float(stats_ready),
        "fea_ratio": float(stats.fea_ratio),
        "cri_ratio": float(stats.cri_ratio),
        "inf_ratio": float(stats.inf_ratio),
        "main_real_cost_rate": float(stats.main_real_cost_rate),
    }
    return float(ratio), info
