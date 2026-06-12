from __future__ import annotations

import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import colorama
except ImportError:  # pragma: no cover
    class _EmptyColors:
        CYAN = GREEN = YELLOW = RESET_ALL = ""

    class _ColoramaFallback:
        Fore = _EmptyColors()
        Style = _EmptyColors()

    colorama = _ColoramaFallback()

from .cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    cfg_get,
    continuous_cost_from_force_prediction,
    dreamer_agent_distribution,
    extract_continuous_cost,
    posterior_states_and_features,
)
from .dual_update import update_dual
from .modules.world_models import predict_force_from_outputs
from .sampling import FDPIRegimeStatsWindow, SourceCostStatsWindow, batch_composition, dual_ratio_from_fdpi_stats
from .trainer_base import (
    OfflineEpisodeWriter,
    _extract_force_obs,
    _is_first,
    _log_info_value,
    _policy_obs,
    _reset_after_step,
)


def _cfg_float(node, name, default):
    return float(cfg_get(node, name, default))


def _cfg_int(node, name, default):
    return int(cfg_get(node, name, default))


def _cfg_bool(node, name, default=False):
    return bool(cfg_get(node, name, default))


def _cfg_int_tuple(node, name, default=()):
    value = cfg_get(node, name, default)
    if value is None:
        return tuple()
    return tuple(int(v) for v in value)


def _node(fdpi_cfg, name):
    return cfg_get(fdpi_cfg, name, None)


def _as_positive_int_list(value, default):
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        values = [int(value)]
    else:
        values = [int(v) for v in value]
    return sorted({v for v in values if v > 0})


class _TrainTimer:
    def __init__(self, device, enabled=False):
        self.device = str(device)
        self.enabled = bool(enabled)
        self.values = {}

    def start(self):
        if not self.enabled:
            return None
        self.sync()
        return time.perf_counter()

    def stop(self, name, token):
        if not self.enabled or token is None:
            return
        self.sync()
        self.values[name] = self.values.get(name, 0.0) + (time.perf_counter() - token)

    def sync(self):
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.synchronize()

    def log(self, logger, step):
        if not self.enabled or logger is None:
            return
        total = sum(self.values.values())
        for name, seconds in self.values.items():
            logger.log(f"Timing/{name}_seconds", float(seconds), step)
        logger.log("Timing/measured_total_seconds", float(total), step)
        self.values.clear()


class _EveryNStepLogger:
    def __init__(self, logger, every_steps=1, always_prefixes=()):
        self.logger = logger
        self.every_steps = max(int(every_steps), 1)
        self.always_prefixes = tuple(always_prefixes)

    def enabled(self, step):
        return int(step) % self.every_steps == 0

    def log(self, tag, value, step):
        if self.logger is None:
            return
        if self.enabled(step) or str(tag).startswith(self.always_prefixes):
            self.logger.log(tag, value, step)

    def log_lazy(self, tag, value_fn, step):
        if self.logger is None:
            return
        if self.enabled(step) or str(tag).startswith(self.always_prefixes):
            self.logger.log(tag, value_fn(), step)

    def log_video(self, tag, value, step):
        if self.logger is not None and self.enabled(step):
            self.logger.log_video(tag, value, step)


class _WorldModelEvalScheduler:
    def __init__(self, *, enabled, start_step, every_steps, initial_step=0):
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.every_steps = max(int(every_steps), 1)
        self.next_step = self.start_step
        initial_step = int(initial_step)
        if initial_step >= self.next_step:
            passed = (initial_step - self.start_step) // self.every_steps + 1
            self.next_step = self.start_step + passed * self.every_steps

    def should_run(self, step):
        return self.enabled and int(step) >= self.next_step

    def mark_ran(self, step):
        step = int(step)
        while self.next_step <= step:
            self.next_step += self.every_steps


def _mean_or_none(value):
    if value is None:
        return None
    value = torch.as_tensor(value)
    if value.numel() == 0:
        return None
    finite = torch.isfinite(value.float())
    if not finite.any():
        return None
    return float(value.float()[finite].mean().detach().cpu().item())


def _average_precision_from_tensors(label, score):
    label = torch.as_tensor(label).detach().float().reshape(-1)
    score = torch.as_tensor(score).detach().float().reshape(-1)
    finite = torch.isfinite(label) & torch.isfinite(score)
    label = label[finite]
    score = score[finite]
    positive = label > 0.5
    num_pos = int(positive.sum().item())
    if label.numel() == 0 or num_pos == 0:
        return None
    order = torch.argsort(score, descending=True)
    sorted_label = positive[order].float()
    tp = torch.cumsum(sorted_label, dim=0)
    fp = torch.cumsum(1.0 - sorted_label, dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / float(num_pos)
    prev_recall = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    return float(((recall - prev_recall) * precision).sum().detach().cpu().item())


def _binary_metrics_from_tensors(label, prob):
    label = torch.as_tensor(label).detach().float().reshape(-1)
    prob = torch.as_tensor(prob).detach().float().reshape(-1)
    finite = torch.isfinite(label) & torch.isfinite(prob)
    label = label[finite]
    prob = prob[finite]
    if label.numel() == 0:
        return {}
    pred = prob >= 0.5
    positive = label > 0.5
    negative = ~positive
    tp = (pred & positive).sum().float()
    fp = (pred & negative).sum().float()
    fn = ((~pred) & positive).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)
    return {
        "auprc": _average_precision_from_tensors(label, prob),
        "precision@0.5": float(precision.detach().cpu().item()),
        "recall@0.5": float(recall.detach().cpu().item()),
        "f1@0.5": float(f1.detach().cpu().item()),
    }


class _WorldModelEvalAccumulator:
    def __init__(self, *, min_samples, high_cost_threshold, boundary_low, boundary_high, cost_splits=True):
        self.min_samples = max(int(min_samples), 1)
        self.high_cost_threshold = float(high_cost_threshold)
        self.boundary_low = float(boundary_low)
        self.boundary_high = float(boundary_high)
        self.cost_splits = bool(cost_splits)
        self.counts = defaultdict(int)
        self.values = defaultdict(lambda: defaultdict(list))
        self.binary = defaultdict(lambda: defaultdict(lambda: {"label": [], "prob": []}))

    def _split_masks(self, source, cost):
        source = torch.as_tensor(source).detach().reshape(-1).to(torch.int64)
        cost = torch.as_tensor(cost).detach().float().reshape(-1)
        masks = {
            "all": torch.ones_like(source, dtype=torch.bool),
            "main": source == SOURCE_MAIN,
            "dual": source == SOURCE_DUAL,
        }
        if self.cost_splits:
            masks["low_cost"] = cost < self.boundary_low
            masks["boundary_cost"] = (cost >= self.boundary_low) & (cost < self.boundary_high)
            masks["high_cost"] = cost >= self.high_cost_threshold
        return masks

    def add_counts(self, source, cost):
        for split, mask in self._split_masks(source, cost).items():
            self.counts[split] += int(mask.sum().item())

    def add_scalar(self, metric, source, cost, value):
        value = torch.as_tensor(value).detach().float().reshape(-1)
        for split, mask in self._split_masks(source, cost).items():
            if mask.numel() != value.numel():
                continue
            selected = value[mask]
            if selected.numel() > 0:
                self.values[split][metric].append(selected.detach().float().cpu())

    def add_binary(self, metric, source, cost, label, prob):
        label = torch.as_tensor(label).detach().float().reshape(-1)
        prob = torch.as_tensor(prob).detach().float().reshape(-1)
        for split, mask in self._split_masks(source, cost).items():
            if mask.numel() != label.numel() or label.numel() != prob.numel():
                continue
            if mask.any():
                store = self.binary[split][metric]
                store["label"].append(label[mask].detach().float().cpu())
                store["prob"].append(prob[mask].detach().float().cpu())

    def summarize(self):
        metrics = {}
        split_metric_values = {}
        for split, count in sorted(self.counts.items()):
            metrics[f"WMEval/{split}/sample_count"] = float(count)
            if count < self.min_samples:
                continue
            split_values = {}
            for name, chunks in self.values.get(split, {}).items():
                if not chunks:
                    continue
                value = _mean_or_none(torch.cat(chunks, dim=0))
                if value is not None:
                    key = f"WMEval/{split}/{name}"
                    metrics[key] = value
                    split_values[name] = value
            for name, chunks in self.binary.get(split, {}).items():
                if not chunks["label"] or not chunks["prob"]:
                    continue
                binary_metrics = _binary_metrics_from_tensors(
                    torch.cat(chunks["label"], dim=0),
                    torch.cat(chunks["prob"], dim=0),
                )
                for suffix, value in binary_metrics.items():
                    if value is not None:
                        key = f"WMEval/{split}/{name}_{suffix}"
                        metrics[key] = value
                        split_values[f"{name}_{suffix}"] = value
            split_metric_values[split] = split_values
        gap_specs = (
            ("prior1_dyn_kl", "prior1/dyn_kl"),
            ("openloop_h10_cost_mae", "openloop_h10/cost_mae"),
            ("openloop_h10_force_mae", "openloop_h10/force_mae"),
            ("openloop_h15_reward_mae", "openloop_h15/reward_mae"),
        )
        main_values = split_metric_values.get("main", {})
        dual_values = split_metric_values.get("dual", {})
        for gap_name, metric_name in gap_specs:
            if metric_name in main_values and metric_name in dual_values:
                metrics[f"WMEvalGap/dual_minus_main/{gap_name}"] = dual_values[metric_name] - main_values[metric_name]
        return metrics


def _sample_training_batches(
    replay_buffer,
    num_batches,
    batch_size,
    horizon,
    *,
    return_dict=True,
    safety_critical_ratio=None,
    high_cost_threshold=0.1,
    boundary_low=0.05,
    boundary_high=0.4,
    use_sample_many=True,
):
    if use_sample_many and hasattr(replay_buffer, "sample_many"):
        return replay_buffer.sample_many(
            int(num_batches),
            batch_size,
            horizon,
            return_dict=return_dict,
            safety_critical_ratio=safety_critical_ratio,
            high_cost_threshold=high_cost_threshold,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
        )
    return [
        replay_buffer.sample(
            batch_size,
            horizon,
            return_dict=return_dict,
            safety_critical_ratio=safety_critical_ratio,
            high_cost_threshold=high_cost_threshold,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
        )
        for _ in range(int(num_batches))
    ]


def train_world_model_step(
    batch,
    world_model,
    agent,
    logger,
    step,
    *,
    compute_detailed_metrics=True,
    return_metrics=True,
):
    if agent is not None:
        agent.eval()
    metrics = world_model.update(
        agent,
        batch["obs"],
        batch["action"],
        batch["reward"],
        batch["done"],
        batch["is_first"],
        force=batch.get("force"),
        cost=batch.get("continuous_cost", batch.get("cost")),
        bottom_force=batch.get("bottom_force"),
        extreme_cost=batch.get("extreme_cost"),
        logger=logger,
        step=step,
        compute_detailed_metrics=compute_detailed_metrics,
        return_metrics=return_metrics,
    )
    if logger is not None and isinstance(metrics, dict):
        if "dyn_loss" in metrics:
            logger.log("WorldModel/dynamics_loss", metrics["dyn_loss"], step)
        if "cost_loss" in metrics:
            logger.log("WorldModel/cost_loss", metrics["cost_loss"], step)
        if "force_loss" in metrics:
            logger.log("WorldModel/force_loss", metrics["force_loss"], step)
        if "force_pred_mean" in metrics:
            logger.log("WorldModel/pred_bottom_force_mean", metrics["force_pred_mean"], step)
        pred_cost = metrics.get("cost/pred_mean", metrics.get("cost/predicted_cost_mean"))
        if isinstance(pred_cost, (int, float)):
            logger.log("WorldModel/pred_cost_mean", pred_cost, step)
    return metrics


def _decode_reward(world_model, deter):
    return world_model.twohot_loss.decode(world_model.reward_head(deter))


def _predict_force_from_feat(world_model, feat):
    if not (getattr(world_model, "force_enabled", False) and getattr(world_model, "force_head", None) is not None):
        return None, None
    flat_feat = feat.flatten(0, 1)
    outputs = world_model.force_head(flat_feat)
    pred, prob = predict_force_from_outputs(
        outputs,
        force_scale=world_model.force_scale,
        threshold=world_model.force_threshold,
        signed_force=world_model.force_signed_force,
    )
    return pred.reshape(*feat.shape[:-1], 1), prob.reshape(*feat.shape[:-1], 1)


def _predict_cost_from_feat(world_model, feat):
    if not hasattr(world_model, "predict_cost"):
        zeros = torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)
        return zeros, zeros
    pred, extreme_prob, _ = world_model.predict_cost(feat)
    return pred, extreme_prob


def _add_prediction_metrics(
    acc,
    prefix,
    *,
    source,
    split_cost,
    reward_target,
    reward_pred,
    cost_target,
    cost_pred,
    extreme_target,
    extreme_prob,
    done_target,
    done_logit,
    force_target=None,
    force_pred=None,
):
    reward_target = torch.as_tensor(reward_target, dtype=reward_pred.dtype, device=reward_pred.device)
    cost_target = torch.as_tensor(cost_target, dtype=cost_pred.dtype, device=cost_pred.device)
    done_target = torch.as_tensor(done_target, dtype=done_logit.dtype, device=done_logit.device)
    extreme_target = torch.as_tensor(extreme_target, dtype=extreme_prob.dtype, device=extreme_prob.device)

    acc.add_scalar(f"{prefix}/reward_mae", source, split_cost, (reward_pred - reward_target).abs())
    acc.add_scalar(f"{prefix}/reward_bias", source, split_cost, reward_pred - reward_target)
    acc.add_scalar(f"{prefix}/cost_mae", source, split_cost, (cost_pred - cost_target).abs())
    acc.add_scalar(f"{prefix}/cost_mse", source, split_cost, (cost_pred - cost_target).pow(2))
    acc.add_scalar(f"{prefix}/cost_bias", source, split_cost, cost_pred - cost_target)
    acc.add_binary(f"{prefix}/extreme", source, split_cost, extreme_target, extreme_prob)
    done_bce = F.binary_cross_entropy_with_logits(done_logit, done_target, reduction="none")
    acc.add_scalar(f"{prefix}/done_bce", source, split_cost, done_bce)

    if force_target is not None and force_pred is not None:
        force_target = torch.as_tensor(force_target, dtype=force_pred.dtype, device=force_pred.device)
        acc.add_scalar(f"{prefix}/force_mae", source, split_cost, (force_pred - force_target).abs())
        acc.add_scalar(f"{prefix}/force_bias", source, split_cost, force_pred - force_target)
        acc.add_scalar(f"{prefix}/force_pred_mean", source, split_cost, force_pred)
        acc.add_scalar(f"{prefix}/force_nonzero_rate", source, split_cost, (force_target.abs() > 1.0e-3).float())


@torch.no_grad()
def run_world_model_eval(
    *,
    replay_buffer,
    world_model,
    logger,
    step,
    cfg,
    num_envs,
    high_cost_threshold,
    boundary_low,
    boundary_high,
    use_sample_many=True,
):
    if logger is None or not _cfg_bool(cfg, "Enable", False):
        return {}

    eval_length = max(_cfg_int(cfg, "EvalLength", 48), 2)
    context_length = max(_cfg_int(cfg, "ContextLength", 16), 1)
    horizons = _as_positive_int_list(cfg_get(cfg, "Horizons", [1, 3, 5, 10, 15, 30]), [1, 3, 5, 10, 15, 30])
    max_horizon = max(horizons) if horizons else 1
    required_length = max(eval_length, context_length + max_horizon + 1)
    if not replay_buffer.can_sample(required_length):
        logger.log("WMEval/skipped_can_sample", 1.0, step)
        return {}

    requested_batch_size = max(_cfg_int(cfg, "BatchSize", 256), int(num_envs))
    eval_batch_size = max(int(num_envs), ((requested_batch_size + int(num_envs) - 1) // int(num_envs)) * int(num_envs))
    num_batches = max(_cfg_int(cfg, "NumBatches", 4), 1)
    min_samples = max(_cfg_int(cfg, "MinSamplesPerSplit", 128), 1)
    cost_splits = _cfg_bool(cfg, "CostSplits", True)

    was_training = world_model.training
    world_model.eval()
    acc = _WorldModelEvalAccumulator(
        min_samples=min_samples,
        high_cost_threshold=high_cost_threshold,
        boundary_low=boundary_low,
        boundary_high=boundary_high,
        cost_splits=cost_splits,
    )

    try:
        batches = _sample_training_batches(
            replay_buffer,
            num_batches,
            eval_batch_size,
            required_length,
            return_dict=True,
            use_sample_many=use_sample_many,
        )
        with torch.autocast(
            device_type=world_model.device_type,
            dtype=world_model.tensor_dtype,
            enabled=world_model.use_amp,
        ):
            for batch in batches:
                obs = batch["obs"].to(world_model.device)
                action = batch["action"].to(world_model.device)
                reward = batch["reward"].to(world_model.device)
                done = batch["done"].to(world_model.device)
                is_first = batch["is_first"].to(world_model.device)
                source = batch["source"].to(world_model.device)
                continuous_cost = batch["continuous_cost"].to(world_model.device)
                extreme_cost = batch["extreme_cost"].to(world_model.device)
                force = batch.get("force")
                if force is not None:
                    force = force.to(world_model.device)

                obs_model = world_model.preprocess(obs)
                post, prior, stoch, deter = world_model.dynamic.parallel_observe(
                    world_model.encoder(obs_model),
                    action,
                    is_first,
                )
                dyn_kl, _, _, _ = world_model.dynamic.kl_loss(post, prior, world_model.kl_free)
                del dyn_kl

                feat = torch.cat((deter, stoch), dim=-1)
                obs_hat = world_model.decoder(stoch)
                reward_pred = _decode_reward(world_model, deter)
                done_logit = world_model.done_head(deter)
                cost_pred, extreme_prob = _predict_cost_from_feat(world_model, feat)
                force_pred, _ = _predict_force_from_feat(world_model, feat)

                posterior_target = {
                    "source": source[:, : feat.shape[1]],
                    "reward": reward[:, : feat.shape[1]],
                    "done": done[:, : feat.shape[1]],
                    "cost": continuous_cost[:, : feat.shape[1]],
                    "extreme": extreme_cost[:, : feat.shape[1]],
                    "force": force[:, : feat.shape[1]] if force is not None else None,
                }
                acc.add_counts(posterior_target["source"], posterior_target["cost"])
                acc.add_scalar(
                    "posterior/recon_mse",
                    posterior_target["source"],
                    posterior_target["cost"],
                    (obs_hat - obs_model[:, : obs_hat.shape[1]]).pow(2).mean(dim=-1, keepdim=True),
                )
                _add_prediction_metrics(
                    acc,
                    "posterior",
                    source=posterior_target["source"],
                    split_cost=posterior_target["cost"],
                    reward_target=posterior_target["reward"],
                    reward_pred=reward_pred,
                    cost_target=posterior_target["cost"],
                    cost_pred=cost_pred,
                    extreme_target=posterior_target["extreme"],
                    extreme_prob=extreme_prob,
                    done_target=posterior_target["done"],
                    done_logit=done_logit,
                    force_target=posterior_target["force"],
                    force_pred=force_pred,
                )

                prior_stoch = world_model.dynamic.get_flatten_stoch(prior)
                prior_feat = torch.cat((prior["deter"], prior_stoch), dim=-1)
                prior_reward = _decode_reward(world_model, prior["deter"])
                prior_done = world_model.done_head(prior["deter"])
                prior_cost, prior_extreme = _predict_cost_from_feat(world_model, prior_feat)
                prior_force, _ = _predict_force_from_feat(world_model, prior_feat)
                post_dist = world_model.dynamic.get_dist(post)
                prior_dist = world_model.dynamic.get_dist(prior)
                prior_dyn_kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist).sum(dim=-1, keepdim=True)
                prior_len = min(prior_reward.shape[1], max(reward.shape[1] - 1, 0))
                if prior_len <= 0:
                    continue
                prior_target = {
                    "source": source[:, 1 : 1 + prior_len],
                    "reward": reward[:, 1 : 1 + prior_len],
                    "done": done[:, 1 : 1 + prior_len],
                    "cost": continuous_cost[:, 1 : 1 + prior_len],
                    "extreme": extreme_cost[:, 1 : 1 + prior_len],
                    "force": force[:, 1 : 1 + prior_len] if force is not None else None,
                }
                prior_reward = prior_reward[:, :prior_len]
                prior_done = prior_done[:, :prior_len]
                prior_cost = prior_cost[:, :prior_len]
                prior_extreme = prior_extreme[:, :prior_len]
                prior_dyn_kl = prior_dyn_kl[:, :prior_len]
                if prior_force is not None:
                    prior_force = prior_force[:, :prior_len]
                acc.add_scalar("prior1/dyn_kl", prior_target["source"], prior_target["cost"], prior_dyn_kl)
                _add_prediction_metrics(
                    acc,
                    "prior1",
                    source=prior_target["source"],
                    split_cost=prior_target["cost"],
                    reward_target=prior_target["reward"],
                    reward_pred=prior_reward,
                    cost_target=prior_target["cost"],
                    cost_pred=prior_cost,
                    extreme_target=prior_target["extreme"],
                    extreme_prob=prior_extreme,
                    done_target=prior_target["done"],
                    done_logit=prior_done,
                    force_target=prior_target["force"],
                    force_pred=prior_force,
                )

                ctx = min(context_length, feat.shape[1] - 1)
                if ctx <= 0:
                    continue
                img_state = {key: value[:, ctx - 1].detach() for key, value in post.items()}
                for horizon in range(1, max_horizon + 1):
                    action_idx = ctx - 1 + horizon
                    if action_idx >= action.shape[1]:
                        break
                    img_state = world_model.dynamic.img_step(img_state, action[:, action_idx])
                    if horizon not in horizons:
                        continue
                    target_idx = ctx + horizon
                    if target_idx >= reward.shape[1]:
                        continue
                    img_deter = world_model.dynamic.get_deter(img_state)[:, None]
                    img_stoch = world_model.dynamic.get_flatten_stoch(img_state)[:, None]
                    img_feat = torch.cat((img_deter, img_stoch), dim=-1)
                    img_reward = _decode_reward(world_model, img_deter)
                    img_done = world_model.done_head(img_deter)
                    img_cost, img_extreme = _predict_cost_from_feat(world_model, img_feat)
                    img_force, _ = _predict_force_from_feat(world_model, img_feat)
                    h_source = source[:, target_idx : target_idx + 1]
                    h_cost = continuous_cost[:, target_idx : target_idx + 1]
                    h_force = force[:, target_idx : target_idx + 1] if force is not None else None
                    _add_prediction_metrics(
                        acc,
                        f"openloop_h{horizon}",
                        source=h_source,
                        split_cost=h_cost,
                        reward_target=reward[:, target_idx : target_idx + 1],
                        reward_pred=img_reward,
                        cost_target=h_cost,
                        cost_pred=img_cost,
                        extreme_target=extreme_cost[:, target_idx : target_idx + 1],
                        extreme_prob=img_extreme,
                        done_target=done[:, target_idx : target_idx + 1],
                        done_logit=img_done,
                        force_target=h_force,
                        force_pred=img_force,
                    )
    finally:
        if was_training:
            world_model.train()
        else:
            world_model.eval()

    metrics = acc.summarize()
    metrics["WMEval/eval_batch_size"] = float(eval_batch_size)
    metrics["WMEval/eval_length"] = float(required_length)
    metrics["WMEval/context_length"] = float(context_length)
    metrics["WMEval/num_batches"] = float(num_batches)
    metrics["WMEval/skipped_can_sample"] = 0.0
    for key, value in metrics.items():
        if value is not None:
            logger.log(key, value, step)
    return metrics


def _predict_imagined_cost(world_model, feat, cost_cfg):
    if hasattr(world_model, "predict_cost"):
        pred_cost, _, _ = world_model.predict_cost(feat)
        return pred_cost.clamp(
            _cfg_float(cost_cfg, "CostMin", 0.0),
            _cfg_float(cost_cfg, "CostMax", 1.0),
        )
    if getattr(world_model, "force_enabled", False) and getattr(world_model, "force_head", None) is not None:
        flat_feat = feat.flatten(0, 1)
        with torch.autocast(device_type=world_model.device_type, dtype=world_model.tensor_dtype, enabled=world_model.use_amp):
            force_outputs = world_model.force_head(flat_feat)
            pred_force, _ = predict_force_from_outputs(
                force_outputs,
                force_scale=world_model.force_scale,
                threshold=world_model.force_threshold,
                signed_force=world_model.force_signed_force,
            )
        pred_force = pred_force.reshape(*feat.shape[:-1], 1)
        return continuous_cost_from_force_prediction(
            pred_force,
            force_threshold=_cfg_float(cost_cfg, "ForceThreshold", 0.1),
            low_force_scale=_cfg_float(cost_cfg, "LowForceScale", 0.05),
            cost_force_max=_cfg_float(cost_cfg, "CostForceMax", 15.0),
            force_scale=_cfg_float(cost_cfg, "ForceScale", 5.0),
            clip_cost=_cfg_bool(cost_cfg, "ClipCost", True),
            cost_min=_cfg_float(cost_cfg, "CostMin", 0.0),
            cost_max=_cfg_float(cost_cfg, "CostMax", 1.0),
        ).to(feat.device)
    return torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)


def _main_fdpi_cfg(fdpi_cfg):
    risk_cfg = _node(fdpi_cfg, "RiskCritic")
    main_cfg = _node(fdpi_cfg, "MainFDPIRegime")
    return {
        "Pf": _cfg_float(risk_cfg, "Pf", 0.40),
        "Cg": _cfg_float(risk_cfg, "Cg", 0.10),
        "RiskMax": _cfg_float(risk_cfg, "RiskMax", 1.0),
        "LambdaCri": _cfg_float(main_cfg, "LambdaCri", 0.001),
        "LambdaInf": _cfg_float(main_cfg, "LambdaInf", 0.002),
        "MinRewardWeightCri": _cfg_float(main_cfg, "MinRewardWeightCri", 0.80),
        "MinRewardWeightInf": _cfg_float(main_cfg, "MinRewardWeightInf", 0.80),
        "EntropyCoef": _cfg_float(main_cfg, "EntropyCoef", 1.0e-4),
        "EntropyCoefFinal": _cfg_float(main_cfg, "EntropyCoefFinal", _cfg_float(main_cfg, "EntropyCoef", 1.0e-4)),
        "EntropyDecayStartStep": _cfg_int(
            main_cfg,
            "EntropyDecayStartStep",
            _cfg_int(main_cfg, "StartStep", 1500000) + _cfg_int(main_cfg, "WarmupSteps", 0),
        ),
        "EntropyDecaySteps": _cfg_int(main_cfg, "EntropyDecaySteps", 0),
        "ActionAnchorCoef": _cfg_float(main_cfg, "ActionAnchorCoef", 0.0),
        "TailRiskCoef": _cfg_float(main_cfg, "TailRiskCoef", 0.0),
        "TailRiskThreshold": _cfg_float(main_cfg, "TailRiskThreshold", _cfg_float(risk_cfg, "Pf", 0.40)),
        "DetachActionForLogProb": _cfg_bool(main_cfg, "DetachActionForLogProb", False),
    }


def train_agent_step(
    samples,
    world_model,
    agent,
    gp_critic,
    imagine_horizon,
    logger,
    step,
    *,
        fdpi_cfg=None,
        compute_fdpi_grad_diagnostics=True,
        return_metrics=True,
):
    world_model.eval()
    feat, action, discount, reward, weight = world_model.imagine_data(
        agent,
        *samples[:5],
        imagine_horizon,
        logger,
        step,
    )
    main_cfg = _node(fdpi_cfg, "MainFDPIRegime")
    if (
        _cfg_bool(main_cfg, "Enable", True)
        and int(step) >= _cfg_int(main_cfg, "StartStep", 1500000)
        and gp_critic is not None
    ):
        info = agent.update_fdpi_regime(
            feat,
            action,
            discount,
            reward,
            weight,
            gp_critic,
            _main_fdpi_cfg(fdpi_cfg),
            logger=logger,
            step=step,
            compute_grad_diagnostics=compute_fdpi_grad_diagnostics,
            return_metrics=return_metrics,
        )
        info["used_fdpi_regime"] = True
        return info
    agent.update(feat, action, discount, reward, weight, logger if return_metrics else None, step)
    if logger is not None:
        logger.log("MainFDPI/enabled", 0.0, step)
    if not return_metrics:
        return {}
    return {
        "used_fdpi_regime": False,
        "task_reward_mean": float(reward.detach().float().mean().item()),
    }


train_world_model_step_dfd_v4 = train_world_model_step
train_agent_step_dfd_v4 = train_agent_step


@torch.no_grad()
def _attach_batched_posterior(world_model, batches, max_batch_size=1024):
    batches = [batch for batch in batches if batch is not None]
    if not batches:
        return
    sizes = [int(batch["obs"].shape[0]) for batch in batches]
    total = sum(sizes)
    if total <= 0:
        return
    obs = torch.cat([batch["obs"].to(world_model.device) for batch in batches], dim=0)
    action = torch.cat([batch["action"].to(world_model.device) for batch in batches], dim=0)
    is_first = torch.cat([batch["is_first"].to(world_model.device) for batch in batches], dim=0)

    max_batch_size = int(max_batch_size or 0)
    state_chunks = []
    feat_chunks = []
    chunk_size = total if max_batch_size <= 0 else max_batch_size
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        post, feat = posterior_states_and_features(
            world_model,
            obs[start:end],
            action[start:end],
            is_first[start:end],
        )
        state_chunks.append({key: value.detach() for key, value in post.items()})
        feat_chunks.append(feat.detach())

    feat_all = torch.cat(feat_chunks, dim=0)
    state_all = {
        key: torch.cat([chunk[key] for chunk in state_chunks], dim=0)
        for key in state_chunks[0].keys()
    }
    cursor = 0
    for batch, size in zip(batches, sizes):
        next_cursor = cursor + size
        batch["_posterior_feat"] = feat_all[cursor:next_cursor]
        batch["_posterior_state"] = {
            key: value[cursor:next_cursor]
            for key, value in state_all.items()
        }
        cursor = next_cursor


def _sample_policy_action(
    *,
    feat,
    agent,
    gp_critic,
    dual_policy,
    world_model,
    state,
    use_dual_sampling,
    dual_ratio,
    num_envs,
    device,
    logger=None,
    step=None,
):
    main_action = agent.sample(feat, greedy=False)
    action = main_action
    source = torch.full((num_envs, 1), SOURCE_MAIN, dtype=torch.int64, device=device)
    g_main = None
    if gp_critic is not None:
        g_main = gp_critic.risk_no_grad(feat, main_action).detach()
    if use_dual_sampling and dual_policy is not None and float(dual_ratio) > 0.0:
        dual_mask = torch.rand(num_envs, device=device) < float(dual_ratio)
        if dual_mask.any():
            dual_action = dual_policy.sample(feat, greedy=False)
            dual_action = dual_action.to(device=main_action.device, dtype=main_action.dtype)
            action = main_action.clone()
            action[dual_mask] = dual_action[dual_mask]
            source[dual_mask] = SOURCE_DUAL
            _log_rollout_dual_action_diagnostics(
                logger,
                step,
                feat=feat,
                agent=agent,
                dual_policy=dual_policy,
                main_action=main_action,
                dual_action=dual_action,
                dual_mask=dual_mask,
            )
    env_action = action.detach().cpu().numpy()
    state = world_model.update_inference_state(state, action)
    return env_action, action, source, state, g_main


def _logger_enabled(logger, step):
    if logger is None or step is None:
        return False
    if hasattr(logger, "enabled") and not logger.enabled(step):
        return False
    return True


def _log_rollout_dual_action_diagnostics(
    logger,
    step,
    *,
    feat,
    agent,
    dual_policy,
    main_action,
    dual_action,
    dual_mask,
):
    if not _logger_enabled(logger, step):
        return
    mask = dual_mask.reshape(-1)
    selected_ratio = mask.float().mean()
    logger.log("DualAction/rollout_selected_ratio", float(selected_ratio.detach().float().item()), step)
    if not bool(mask.any().item()):
        return

    with torch.no_grad():
        selected_feat = feat[mask]
        selected_main_action = main_action[mask]
        selected_dual_action = dual_action[mask]
        sample_gap = selected_dual_action - selected_main_action
        dual_dist = dual_policy.distribution(selected_feat)
        main_dist = dreamer_agent_distribution(agent, selected_feat)
        mean_gap = dual_dist.base_dist.loc.detach() - main_dist.base_dist.loc.detach()
        dual_logprob = dual_dist.log_prob(selected_dual_action)[..., None]
        main_logprob = main_dist.log_prob(selected_dual_action)[..., None]

        metrics = {
            "sample_l2": sample_gap.float().pow(2).sum(dim=-1).sqrt().mean(),
            "sample_abs_mean": sample_gap.float().abs().mean(),
            "mean_l2": mean_gap.float().pow(2).sum(dim=-1).sqrt().mean(),
            "mean_abs_mean": mean_gap.float().abs().mean(),
            "dual_logprob_on_dual": dual_logprob.float().mean(),
            "main_logprob_on_dual": main_logprob.float().mean(),
            "logprob_gap": (dual_logprob - main_logprob).float().mean(),
        }
    for name, value in metrics.items():
        logger.log(f"DualAction/rollout_{name}", float(value.detach().float().item()), step)


def _action_bounds(vec_env, device, dtype):
    space = getattr(vec_env, "single_action_space", getattr(vec_env, "action_space", None))
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None, None
    low_t = torch.as_tensor(low, dtype=dtype, device=device).reshape(1, -1)
    high_t = torch.as_tensor(high, dtype=dtype, device=device).reshape(1, -1)
    return low_t, high_t


def _sample_warmup_policy_noise_action(
    *,
    current_obs,
    is_first,
    agent,
    world_model,
    state,
    vec_env,
    num_envs,
    device,
    noise_std,
    greedy_base=False,
):
    with torch.no_grad():
        world_model.eval()
        agent.eval()
        feat, state = world_model.get_inference_feat(state, current_obs, is_first)
        base_action = agent.sample(feat, greedy=bool(greedy_base))
        if float(noise_std) > 0.0:
            noise = torch.randn_like(base_action) * float(noise_std)
            action = base_action + noise
        else:
            action = base_action
        low, high = _action_bounds(vec_env, device, action.dtype)
        if low is not None and high is not None:
            action = torch.max(torch.min(action, high), low)
        else:
            action = action.clamp(-1.0, 1.0)
        source = torch.full((num_envs, 1), SOURCE_MAIN, dtype=torch.int64, device=device)
        state = world_model.update_inference_state(state, action)
        env_action = action.detach().cpu().numpy()
    return env_action, action, source, state


def _log_replay_stats(replay_buffer, logger, step, *, high_cost_threshold=0.1, boundary_low=0.05, boundary_high=0.4):
    if not hasattr(replay_buffer, "source_stats"):
        return
    stats = replay_buffer.source_stats()
    total = max(sum(stats.values()), 1)
    logger.log("Replay/source_main_ratio", stats.get("main", 0) / total, step)
    logger.log("Replay/source_dual_ratio", stats.get("dual", 0) / total, step)
    logger.log("Replay/source_random_ratio", stats.get("random", 0) / total, step)
    if hasattr(replay_buffer, "cost_stats"):
        cost_stats = replay_buffer.cost_stats(
            high_cost_threshold=high_cost_threshold,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
        )
        for key, value in cost_stats.items():
            logger.log(f"Replay/{key}", value, step)


def _log_batch_composition(logger, prefix, batch, step, *, high_cost_threshold, boundary_low, boundary_high):
    stats = batch_composition(
        batch,
        high_cost_threshold=high_cost_threshold,
        boundary_low=boundary_low,
        boundary_high=boundary_high,
    )
    for key, value in stats.items():
        logger.log(f"{prefix}/{key}", value, step)


_INFO_LOG_KEYWORDS = ("reward", "force")
_INFO_SKIP_KEYS = {"terminal_observation"}


def _should_log_info_key(path):
    lower_path = str(path).lower()
    return any(keyword in lower_path for keyword in _INFO_LOG_KEYWORDS)


def _numeric_numpy_array(value):
    if not isinstance(value, np.ndarray):
        return False
    return np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)


def _mask_info_value(value, mask):
    if value is None:
        return None

    if isinstance(value, dict):
        masked = {}
        for key, sub_value in value.items():
            sub_masked = _mask_info_value(sub_value, mask)
            if sub_masked is not None:
                masked[key] = sub_masked
        return masked or None

    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] != mask.numel():
            return None
        selected = mask.to(device=value.device)
        if not bool(selected.any().item()):
            return None
        return value.detach()[selected]

    if isinstance(value, (list, tuple)):
        try:
            value = np.asarray(value)
        except Exception:
            return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != mask.numel() or not _numeric_numpy_array(value):
            return None
        selected = mask.detach().cpu().numpy().astype(bool)
        if not bool(selected.any()):
            return None
        return value[selected]

    return None


def _log_reward_force_info_value(logger, tag, value, step):
    if value is None:
        return

    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in _INFO_SKIP_KEYS:
                continue
            _log_reward_force_info_value(logger, f"{tag}/{key}", sub_value, step)
        return

    if _should_log_info_key(tag):
        _log_info_value(logger, tag, value, step)


def _log_reward_force_info_value_by_source(logger, tag, value, source_masks, step):
    if value is None:
        return

    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in _INFO_SKIP_KEYS:
                continue
            _log_reward_force_info_value_by_source(logger, f"{tag}/{key}", sub_value, source_masks, step)
        return

    if not _should_log_info_key(tag):
        return

    for prefix, mask in source_masks:
        masked_value = _mask_info_value(value, mask)
        if masked_value is not None:
            _log_info_value(logger, f"{prefix}/{tag}", masked_value, step)


def _log_info_dict_reward_force_by_source(logger, info, source, step):
    if not isinstance(info, dict):
        return
    source_mask = source.detach().reshape(-1)
    source_masks = (
        ("InfoMain", source_mask == SOURCE_MAIN),
        ("InfoDual", source_mask == SOURCE_DUAL),
    )
    for key, value in info.items():
        if key in _INFO_SKIP_KEYS:
            continue
        _log_reward_force_info_value(logger, f"Info/{key}", value, step)
        _log_reward_force_info_value_by_source(logger, key, value, source_masks, step)


def _extract_left_right_bottom_force(obs_dict, *, num_envs, device, force_key="", bottom_force_channels=(2, 5)):
    if not isinstance(obs_dict, dict):
        return None, None
    candidate_keys = tuple(key for key in (force_key, "force") if key)
    for key in candidate_keys:
        value = obs_dict.get(key)
        if value is None:
            continue
        force = torch.as_tensor(value, dtype=torch.float32, device=device)
        if force.ndim == 0 or force.shape[0] != num_envs:
            continue
        force = torch.nan_to_num(force.reshape(num_envs, -1).abs(), nan=0.0, posinf=1.0e6)
        if len(bottom_force_channels) >= 2 and force.shape[-1] > max(bottom_force_channels[:2]):
            left_idx = int(bottom_force_channels[0])
            right_idx = int(bottom_force_channels[1])
            return force[:, left_idx], force[:, right_idx]
    return None, None


def _log_bottom_side_info_by_source(logger, source, leftbottom, rightbottom, step):
    _log_side_force_info_by_source(logger, source, leftbottom, rightbottom, step, "leftbottom", "rightbottom")


def _log_wall_side_info_by_source(logger, source, leftwall, rightwall, step):
    _log_side_force_info_by_source(logger, source, leftwall, rightwall, step, "leftwall", "rightwall")


def _log_side_force_info_by_source(logger, source, left_force, right_force, step, left_key, right_key):
    source_mask = source.detach().reshape(-1)
    for prefix, mask in (("InfoMain", source_mask == SOURCE_MAIN), ("InfoDual", source_mask == SOURCE_DUAL)):
        mask = mask.to(device=source.device)
        if not bool(mask.any().item()):
            continue
        if left_force is not None:
            logger.log(f"{prefix}/{left_key}", left_force.to(device=source.device)[mask].float().mean().item(), step)
        if right_force is not None:
            logger.log(f"{prefix}/{right_key}", right_force.to(device=source.device)[mask].float().mean().item(), step)


def _module_optimizer_state(module):
    optimizer = getattr(module, "optimizer", None)
    return optimizer.state_dict() if optimizer is not None else None


def _module_scaler_state(module):
    scaler = getattr(module, "scaler", None)
    return scaler.state_dict() if scaler is not None else None


def _agent_ema_state(agent):
    state = {}
    for name in ("lower_ema", "upper_ema"):
        ema = getattr(agent, name, None)
        if ema is not None:
            state[name] = {
                "scalar": float(getattr(ema, "scalar", 0.0)),
                "decay": float(getattr(ema, "decay", 0.0)),
            }
    return state


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _save_full_state(
    path,
    *,
    env_steps,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    replay_buffer,
    save_replay_buffer=True,
    save_optimizer=True,
):
    payload = {
        "format": "fdpi_reachability_dreamer_isaaclab22_full_state",
        "version": 1,
        "env_steps": int(env_steps),
        "world_model_state_dict": world_model.state_dict(),
        "agent_state_dict": agent.state_dict(),
        "gp_state_dict": gp_critic.state_dict(),
        "gd_state_dict": gd_critic.state_dict(),
        "dual_policy_state_dict": dual_policy.state_dict(),
        "agent_ema_state": _agent_ema_state(agent),
        "rng_state": _rng_state(),
    }
    if save_optimizer:
        payload["optimizer_state_dicts"] = {
            "world_model": _module_optimizer_state(world_model),
            "agent": _module_optimizer_state(agent),
            "gp": _module_optimizer_state(gp_critic),
            "gd": _module_optimizer_state(gd_critic),
            "dual_policy": _module_optimizer_state(dual_policy),
        }
        payload["scaler_state_dicts"] = {
            "world_model": _module_scaler_state(world_model),
            "agent": _module_scaler_state(agent),
        }
    if save_replay_buffer:
        payload["replay_buffer_state_dict"] = replay_buffer.state_dict(cpu=True)
    torch.save(payload, path)


def joint_train_fdpi(
    env_name,
    run_name,
    vec_env,
    max_steps,
    replay_buffer,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    fdpi_cfg,
    train_model_every_steps,
    train_agent_every_steps,
    model_update,
    agent_update,
    batch_size,
    batch_length,
    imagine_batch_size,
    imagine_context,
    imagine_horizon,
    save_every_steps,
    logger,
    device,
    offline_dataset_dir=None,
    checkpoint_dir=None,
    initial_env_steps=0,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or f"ckpt/{run_name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(colorama.Fore.CYAN + f"Saving FDPI reachability Dreamer checkpoints to {checkpoint_dir}" + colorama.Style.RESET_ALL)

    num_envs = vec_env.num_envs
    offline_episode_writer = None
    model_update = max(int(model_update), 1)
    agent_update = max(int(agent_update), 1)
    batch_size = int(batch_size)
    batch_length = int(batch_length)
    imagine_batch_size = int(imagine_batch_size)
    imagine_context = int(imagine_context)
    if imagine_batch_size <= 0:
        imagine_batch_size = batch_size
    if imagine_context <= 0:
        imagine_context = batch_length

    replay_cfg = _node(fdpi_cfg, "Replay")
    warmup_sampling_cfg = _node(fdpi_cfg, "WarmupSampling")
    cost_cfg = _node(fdpi_cfg, "ContinuousCost")
    risk_cfg = _node(fdpi_cfg, "RiskCritic")
    gp_cfg = _node(fdpi_cfg, "Gp")
    gd_cfg = _node(fdpi_cfg, "Gd")
    dual_update_cfg = _node(fdpi_cfg, "DualUpdate")
    dual_sampling_cfg = _node(fdpi_cfg, "DualSampling")
    wm_sampling_cfg = _node(fdpi_cfg, "WorldModelSampling")
    wm_eval_cfg = _node(fdpi_cfg, "WorldModelEval")
    checkpoint_cfg = _node(fdpi_cfg, "Checkpoint")
    performance_cfg = _node(fdpi_cfg, "Performance")

    high_cost_threshold = _cfg_float(wm_sampling_cfg, "HighCostThreshold", _cfg_float(gp_cfg, "HighCostThreshold", 0.1))
    boundary_low = _cfg_float(wm_sampling_cfg, "BoundaryLow", _cfg_float(gp_cfg, "BoundaryLow", 0.05))
    boundary_high = _cfg_float(wm_sampling_cfg, "BoundaryHigh", _cfg_float(gp_cfg, "BoundaryHigh", 0.4))
    world_model_safety_ratio = (
        _cfg_float(wm_sampling_cfg, "SafetyCriticalRatio", 0.20)
        if _cfg_bool(wm_sampling_cfg, "EnableSafetyCriticalSampling", True)
        else 0.0
    )
    cost_source = str(cfg_get(cost_cfg, "CostSource", "bottom"))
    bottom_channels = _cfg_int_tuple(cost_cfg, "BottomForceChannels", [2, 5])
    wall_channels = _cfg_int_tuple(cost_cfg, "WallForceChannels", [1, 4])
    explicit_cost_channels = _cfg_int_tuple(cost_cfg, "CostForceChannels", ())
    cost_force_channels = explicit_cost_channels if explicit_cost_channels else None
    warmup_noise_std = _cfg_float(warmup_sampling_cfg, "NoiseStd", 0.50)
    warmup_greedy_base = _cfg_bool(warmup_sampling_cfg, "GreedyBase", False)
    save_full_state = _cfg_bool(checkpoint_cfg, "SaveFullState", True)
    save_replay_buffer = _cfg_bool(checkpoint_cfg, "SaveReplayBuffer", True)
    save_optimizer = _cfg_bool(checkpoint_cfg, "SaveOptimizer", True)
    full_state_prefix = str(cfg_get(checkpoint_cfg, "FullStatePrefix", "full_state"))
    gp_update_steps = max(_cfg_int(gp_cfg, "UpdateSteps", 1), 1)
    gd_update_steps = max(_cfg_int(gd_cfg, "UpdateSteps", 1), 1)
    gd_start_step = _cfg_int(gd_cfg, "StartStep", 0)
    dual_update_steps = max(_cfg_int(dual_update_cfg, "UpdateSteps", 1), 1)
    log_every_steps = max(_cfg_int(performance_cfg, "LogEverySteps", 1), 1)
    detailed_log_every_steps = max(_cfg_int(performance_cfg, "DetailedLogEverySteps", log_every_steps), 1)
    timing_log_every_steps = max(_cfg_int(performance_cfg, "TimingLogEverySteps", 0), 0)
    use_sample_many = _cfg_bool(performance_cfg, "UseSampleMany", True)
    use_batched_critic_latent = _cfg_bool(performance_cfg, "UseBatchedCriticLatentEncoding", True)
    batched_latent_max_batch = max(_cfg_int(performance_cfg, "BatchedLatentEncodeMaxBatch", 1024), 0)
    fdpi_grad_diagnostics_every_steps = _cfg_int(performance_cfg, "FDPIGradDiagnosticsEverySteps", 65536)
    timing_enabled = timing_log_every_steps > 0
    gp_enabled = _cfg_bool(gp_cfg, "Enable", True)
    step_logger = _EveryNStepLogger(
        logger,
        every_steps=log_every_steps,
        always_prefixes=("Rollout/IsaacLab/",),
    )
    detailed_logger = _EveryNStepLogger(logger, every_steps=detailed_log_every_steps)
    if hasattr(replay_buffer, "cache_starts"):
        replay_buffer.cache_starts = _cfg_bool(performance_cfg, "CacheReplayStarts", True)
    pf = _cfg_float(risk_cfg, "Pf", 0.10)
    cg = _cfg_float(risk_cfg, "Cg", 0.03)
    recent_cost_window_len = _cfg_int(dual_sampling_cfg, "FeasibleRatioWindow", 10000)
    feasible_window = FDPIRegimeStatsWindow(recent_cost_window_len)
    recent_cost_window = SourceCostStatsWindow(recent_cost_window_len)

    model_update_count = 0
    agent_update_count = 0
    gp_update_count = 0
    gd_update_count = 0
    dual_update_count = 0
    last_dual_kl = 0.0

    episode_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_cost = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_bottom_force_peak = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_len = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if offline_dataset_dir:
        offline_episode_writer = OfflineEpisodeWriter(offline_dataset_dir, num_envs)
        print(colorama.Fore.CYAN + f"Saving offline episodes to {offline_episode_writer.output_dir}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()
    gp_critic.eval()
    gd_critic.eval()
    dual_policy.eval()
    state = world_model.initial(num_envs)
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(device)
    is_first = _is_first(current_obs_dict, num_envs, device)
    episodes_completed = 0
    episode_successes = 0
    episode_failures = 0
    episode_timeouts = 0

    initial_env_steps = max(int(initial_env_steps), 0)
    logger.log(f"Rollout/IsaacLab/{env_name}_reward", 0, initial_env_steps)
    logger.log("Rollout/buffer_length", 0, initial_env_steps)
    remaining_steps = max(int(max_steps) - initial_env_steps, 0)
    total_iters = remaining_steps // num_envs
    train_model_every_iters = max(train_model_every_steps // num_envs, 1)
    train_agent_every_iters = max(train_agent_every_steps // num_envs, 1)
    save_every_iters = max(save_every_steps // num_envs, 1)
    wm_eval_scheduler = _WorldModelEvalScheduler(
        enabled=_cfg_bool(wm_eval_cfg, "Enable", False),
        start_step=_cfg_int(wm_eval_cfg, "StartStep", 1000000),
        every_steps=_cfg_int(wm_eval_cfg, "EverySteps", 1000000),
        initial_step=initial_env_steps,
    )

    for iter_idx in tqdm(range(total_iters)):
        env_steps = initial_env_steps + iter_idx * num_envs
        g_main_for_window = None
        timer = _TrainTimer(device, enabled=timing_enabled and env_steps % timing_log_every_steps == 0)

        if replay_buffer.ready():
            token = timer.start()
            with torch.no_grad():
                world_model.eval()
                agent.eval()
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                stats = feasible_window.stats()
                dual_ratio, ratio_info = dual_ratio_from_fdpi_stats(
                    step=env_steps,
                    cfg=dual_sampling_cfg,
                    stats=stats,
                    last_dual_kl=last_dual_kl,
                )
                env_action, action, source, state, g_main_for_window = _sample_policy_action(
                    feat=feat,
                    agent=agent,
                    gp_critic=gp_critic if gp_enabled else None,
                    dual_policy=dual_policy,
                    world_model=world_model,
                    state=state,
                    use_dual_sampling=dual_ratio > 0.0,
                    dual_ratio=dual_ratio,
                    num_envs=num_envs,
                    device=device,
                    logger=step_logger,
                    step=env_steps,
                )
                step_logger.log("Dual/ratio", dual_ratio, env_steps)
                step_logger.log("Dual/active", float(dual_ratio > 0.0), env_steps)
                step_logger.log("Dual/kl_to_main", float(last_dual_kl), env_steps)
                for key, value in ratio_info.items():
                    detailed_logger.log(f"DualSampling/{key}", value, env_steps)
            timer.stop("policy_inference", token)
        else:
            token = timer.start()
            env_action, action, source, state = _sample_warmup_policy_noise_action(
                current_obs=current_obs,
                is_first=is_first,
                agent=agent,
                world_model=world_model,
                state=state,
                vec_env=vec_env,
                num_envs=num_envs,
                device=device,
                noise_std=warmup_noise_std,
                greedy_base=warmup_greedy_base,
            )
            timer.stop("policy_inference", token)
            step_logger.log("Warmup/policy_noise", 1.0, env_steps)
            step_logger.log("Warmup/noise_std", warmup_noise_std, env_steps)

        token = timer.start()
        next_obs_dict, reward, done, info = vec_env.step(env_action)
        timer.stop("env_step", token)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        info_for_log = dict(info) if isinstance(info, dict) else {}
        info_for_log.setdefault("reward", reward)
        if detailed_logger.enabled(env_steps):
            _log_info_dict_reward_force_by_source(logger, info_for_log, source, env_steps)

        token = timer.start()
        cost_parts = extract_continuous_cost(
            info,
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_threshold=_cfg_float(cost_cfg, "ForceThreshold", 0.1),
            low_force_scale=_cfg_float(cost_cfg, "LowForceScale", 0.05),
            cost_force_max=_cfg_float(cost_cfg, "CostForceMax", 15.0),
            force_scale=_cfg_float(cost_cfg, "ForceScale", 5.0),
            extreme_force_threshold=_cfg_float(cost_cfg, "ExtremeForceThreshold", 5.0),
            clip_cost=_cfg_bool(cost_cfg, "ClipCost", True),
            cost_min=_cfg_float(cost_cfg, "CostMin", 0.0),
            cost_max=_cfg_float(cost_cfg, "CostMax", 1.0),
            force_key=getattr(replay_buffer, "force_key", ""),
            cost_source=cost_source,
            bottom_force_channels=bottom_channels,
            wall_force_channels=wall_channels,
            cost_force_channels=cost_force_channels,
        )
        continuous_cost = cost_parts["continuous_cost"]
        binary_cost = cost_parts["binary_cost"]
        extreme_cost = cost_parts["extreme_cost"]
        bottom_force = cost_parts["bottom_force"]
        force_excess = cost_parts["force_excess"]
        leftbottom, rightbottom = _extract_left_right_bottom_force(
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_key=getattr(replay_buffer, "force_key", ""),
            bottom_force_channels=bottom_channels,
        )
        if detailed_logger.enabled(env_steps):
            _log_bottom_side_info_by_source(logger, source, leftbottom, rightbottom, env_steps)
        leftwall, rightwall = _extract_left_right_bottom_force(
            next_obs_dict,
            num_envs=num_envs,
            device=device,
            force_key=getattr(replay_buffer, "force_key", ""),
            bottom_force_channels=wall_channels,
        )
        if detailed_logger.enabled(env_steps):
            _log_wall_side_info_by_source(logger, source, leftwall, rightwall, env_steps)

        if g_main_for_window is not None:
            feasible_window.append(
                g_main=g_main_for_window,
                source=source,
                continuous_cost=continuous_cost,
                pf=pf,
                cg=cg,
            )
        recent_cost_window.append(
            source=source,
            continuous_cost=continuous_cost,
            binary_cost=binary_cost,
            extreme_cost=extreme_cost,
        )

        terminal = torch.as_tensor(
            next_obs_dict.get("is_terminal", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        failure = torch.as_tensor(
            next_obs_dict.get("failure", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        episode_success = info.get("episode_success")
        if episode_success is None:
            episode_success = terminal & ~failure
        else:
            episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=device).view(-1)
        episode_failure = info.get("episode_failure")
        if episode_failure is None:
            episode_failure = terminal & failure
        else:
            episode_failure = torch.as_tensor(episode_failure, dtype=torch.bool, device=device).view(-1)
        episode_timeout = info.get("episode_timeout")
        if episode_timeout is None:
            episode_timeout = done & ~terminal
        else:
            episode_timeout = torch.as_tensor(episode_timeout, dtype=torch.bool, device=device).view(-1)

        force = None
        if getattr(replay_buffer, "include_force", False):
            force = _extract_force_obs(
                current_obs_dict,
                num_envs,
                device,
                getattr(replay_buffer, "force_key", ""),
            )

        replay_buffer.append(
            current_obs,
            action,
            reward,
            done,
            is_first,
            force=force,
            continuous_cost=continuous_cost,
            binary_cost=binary_cost,
            extreme_cost=extreme_cost,
            bottom_force=bottom_force,
            force_excess=force_excess,
            source=source,
        )
        timer.stop("replay_append", token)
        if offline_episode_writer is not None:
            offline_episode_writer.append_step(current_obs_dict, action, reward, done, is_first, env_steps + num_envs)

        episode_reward += reward
        episode_cost += continuous_cost.view(-1)
        episode_bottom_force += bottom_force.view(-1)
        episode_bottom_force_peak = torch.maximum(episode_bottom_force_peak, bottom_force.view(-1))
        episode_len += 1.0
        step_logger.log_lazy("Main/continuous_cost_mean", lambda: continuous_cost.float().mean().item(), env_steps)
        step_logger.log_lazy("Dual/source_count", lambda: float((source == SOURCE_DUAL).sum().item()), env_steps)
        if step_logger.enabled(env_steps):
            for key, value in recent_cost_window.stats().items():
                logger.log(f"RecentCost/{key}", value, env_steps)
        if step_logger.enabled(env_steps) and bool((source == SOURCE_DUAL).any().item()):
            step_logger.log(
                "Dual/real_cost_mean",
                continuous_cost[source.view(-1) == SOURCE_DUAL].float().mean().item(),
                env_steps,
            )

        if done.any():
            token = timer.start()
            done_indices = torch.nonzero(done, as_tuple=False).flatten()
            completed_now = int(done_indices.numel())
            success_now = int(episode_success[done_indices].sum().item())
            failure_now = int(episode_failure[done_indices].sum().item())
            timeout_now = int(episode_timeout[done_indices].sum().item())

            episodes_completed += completed_now
            episode_successes += success_now
            episode_failures += failure_now
            episode_timeouts += timeout_now

            for idx in done_indices.tolist():
                ep_len = max(float(episode_len[idx].item()), 1.0)
                if replay_buffer.ready():
                    logger.log(f"Rollout/IsaacLab/{env_name}_reward", episode_reward[idx].item(), env_steps)
                    step_logger.log("Rollout/episode_cost", episode_cost[idx].item(), env_steps)
                    logger.log("Rollout/buffer_length", len(replay_buffer), env_steps)
                    step_logger.log("Main/task_return", episode_reward[idx].item(), env_steps)
                    step_logger.log("Main/episode_cost_mean", episode_cost[idx].item() / ep_len, env_steps)
                    detailed_logger.log("Main/bottom_force_mean", episode_bottom_force[idx].item() / ep_len, env_steps)
                    detailed_logger.log("Main/bottom_force_peak", episode_bottom_force_peak[idx].item(), env_steps)
                    step_logger.log("Main/success_rate", episode_successes / max(episodes_completed, 1), env_steps)
                episode_reward[idx] = 0.0
                episode_cost[idx] = 0.0
                episode_bottom_force[idx] = 0.0
                episode_bottom_force_peak[idx] = 0.0
                episode_len[idx] = 0.0

            logger.log("Rollout/episodes_completed", episodes_completed, env_steps)
            step_logger.log("Rollout/episode_successes", episode_successes, env_steps)
            step_logger.log("Rollout/episode_failures", episode_failures, env_steps)
            step_logger.log("Rollout/episode_timeouts", episode_timeouts, env_steps)
            logger.log("Rollout/episode_success_rate", episode_successes / max(episodes_completed, 1), env_steps)
            step_logger.log("Rollout/episode_failure_rate", episode_failures / max(episodes_completed, 1), env_steps)
            step_logger.log("Rollout/episode_timeout_rate", episode_timeouts / max(episodes_completed, 1), env_steps)
            timer.stop("logging", token)

        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, device)

        if replay_buffer.ready():
            if iter_idx % train_model_every_iters == 0 and replay_buffer.can_sample(batch_length):
                detailed_metrics_enabled = detailed_logger.enabled(env_steps)
                token = timer.start()
                batches = _sample_training_batches(
                    replay_buffer,
                    model_update,
                    batch_size,
                    batch_length,
                    return_dict=True,
                    safety_critical_ratio=world_model_safety_ratio,
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                    use_sample_many=use_sample_many,
                )
                timer.stop("sample_world_model_batch", token)
                for batch in batches:
                    if detailed_metrics_enabled:
                        _log_batch_composition(
                            logger,
                            "WorldModelBatch",
                            batch,
                            env_steps,
                            high_cost_threshold=high_cost_threshold,
                            boundary_low=boundary_low,
                            boundary_high=boundary_high,
                        )
                    token = timer.start()
                    train_world_model_step(
                        batch,
                        world_model,
                        agent,
                        detailed_logger if detailed_metrics_enabled else None,
                        env_steps,
                        compute_detailed_metrics=detailed_metrics_enabled,
                        return_metrics=detailed_metrics_enabled,
                    )
                    timer.stop("world_model_update", token)
                    model_update_count += 1

            if wm_eval_scheduler.should_run(env_steps):
                token = timer.start()
                run_world_model_eval(
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    logger=logger,
                    step=env_steps,
                    cfg=wm_eval_cfg,
                    num_envs=num_envs,
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                    use_sample_many=use_sample_many,
                )
                timer.stop("world_model_eval", token)
                wm_eval_scheduler.mark_ran(env_steps)

            gp_batches = []
            gd_batches = []
            dual_batches = []
            should_train_agent_side = iter_idx % train_agent_every_iters == 0
            detailed_metrics_enabled = detailed_logger.enabled(env_steps)
            policy_metrics_enabled = step_logger.enabled(env_steps)

            if gp_enabled and should_train_agent_side and replay_buffer.can_sample(batch_length):
                token = timer.start()
                gp_batches = _sample_training_batches(
                    replay_buffer,
                    gp_update_steps,
                    batch_size,
                    batch_length,
                    return_dict=True,
                    safety_critical_ratio=_cfg_float(gp_cfg, "SafetyCriticalRatio", 0.20),
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                    use_sample_many=use_sample_many,
                )
                timer.stop("sample_gp_batch", token)

            if (
                _cfg_bool(gd_cfg, "Enable", True)
                and env_steps >= gd_start_step
                and should_train_agent_side
                and replay_buffer.can_sample(batch_length)
            ):
                token = timer.start()
                gd_batches = _sample_training_batches(
                    replay_buffer,
                    gd_update_steps,
                    batch_size,
                    batch_length,
                    return_dict=True,
                    safety_critical_ratio=_cfg_float(gd_cfg, "SafetyCriticalRatio", 0.40),
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                    use_sample_many=use_sample_many,
                )
                timer.stop("sample_gd_batch", token)

            if (
                _cfg_bool(dual_update_cfg, "Enable", True)
                and env_steps >= _cfg_int(dual_update_cfg, "StartStep", 100000)
                and should_train_agent_side
                and replay_buffer.can_sample(batch_length)
            ):
                token = timer.start()
                dual_batches = _sample_training_batches(
                    replay_buffer,
                    dual_update_steps,
                    batch_size,
                    batch_length,
                    return_dict=True,
                    use_sample_many=use_sample_many,
                )
                timer.stop("sample_dual_batch", token)

            if use_batched_critic_latent and (gp_batches or gd_batches or dual_batches):
                token = timer.start()
                _attach_batched_posterior(
                    world_model,
                    [*gp_batches, *gd_batches, *dual_batches],
                    max_batch_size=batched_latent_max_batch,
                )
                timer.stop("critic_dual_latent_encode", token)

            for batch in gp_batches:
                token = timer.start()
                gp_critic.update(
                    batch,
                    world_model,
                    agent,
                    dual_policy,
                    logger=detailed_logger if detailed_metrics_enabled else None,
                    step=env_steps,
                    posterior_feat=batch.get("_posterior_feat") if use_batched_critic_latent else None,
                    return_metrics=detailed_metrics_enabled,
                )
                timer.stop("gp_update", token)
                gp_update_count += 1

            for batch in gd_batches:
                token = timer.start()
                gd_critic.update(
                    batch,
                    world_model,
                    dual_policy,
                    logger=detailed_logger if detailed_metrics_enabled else None,
                    step=env_steps,
                    posterior_feat=batch.get("_posterior_feat") if use_batched_critic_latent else None,
                    return_metrics=detailed_metrics_enabled,
                )
                timer.stop("gd_update", token)
                gd_update_count += 1

            for batch in dual_batches:
                token = timer.start()
                info_dual = update_dual(
                    batch,
                    world_model,
                    agent,
                    gd_critic,
                    dual_policy,
                    dual_update_cfg,
                    cost_cfg=cost_cfg,
                    logger=step_logger if policy_metrics_enabled else None,
                    step=env_steps,
                    posterior_state=batch.get("_posterior_state") if use_batched_critic_latent else None,
                    return_metrics=policy_metrics_enabled,
                )
                timer.stop("dual_update", token)
                last_dual_kl = abs(float(info_dual.get("kl_to_main", 0.0))) if info_dual else last_dual_kl
                dual_update_count += 1

            if should_train_agent_side and replay_buffer.can_sample(imagine_context):
                token = timer.start()
                imagine_batches = _sample_training_batches(
                    replay_buffer,
                    agent_update,
                    imagine_batch_size,
                    imagine_context,
                    return_dict=False,
                    use_sample_many=use_sample_many,
                )
                timer.stop("sample_agent_batch", token)
                for imagine_samples in imagine_batches:
                    token = timer.start()
                    train_agent_step(
                        imagine_samples,
                        world_model,
                        agent,
                        gp_critic if gp_enabled else None,
                        imagine_horizon,
                        step_logger if policy_metrics_enabled else None,
                        env_steps,
                        fdpi_cfg=fdpi_cfg,
                        compute_fdpi_grad_diagnostics=(
                            fdpi_grad_diagnostics_every_steps > 0
                            and env_steps % fdpi_grad_diagnostics_every_steps == 0
                            and detailed_metrics_enabled
                        ),
                        return_metrics=policy_metrics_enabled,
                    )
                    timer.stop("agent_update", token)
                    agent_update_count += 1

            token = timer.start()
            collected_steps = env_steps + num_envs
            step_logger.log("Train/model_updates", model_update_count, env_steps)
            step_logger.log("Train/agent_updates", agent_update_count, env_steps)
            step_logger.log("Train/gp_updates", gp_update_count, env_steps)
            step_logger.log("Train/gd_updates", gd_update_count, env_steps)
            step_logger.log("Train/dual_updates", dual_update_count, env_steps)
            step_logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
            step_logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)
            if detailed_metrics_enabled:
                _log_replay_stats(
                    replay_buffer,
                    logger,
                    env_steps,
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                )
            timer.stop("logging", token)

        if iter_idx % save_every_iters == 0:
            token = timer.start()
            print(colorama.Fore.GREEN + f"Saving FDPI reachability Dreamer model at total steps {env_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{env_steps}.pth"))
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_{env_steps}.pth"))
            torch.save(gp_critic.state_dict(), os.path.join(checkpoint_dir, f"gp_{env_steps}.pth"))
            torch.save(gd_critic.state_dict(), os.path.join(checkpoint_dir, f"gd_{env_steps}.pth"))
            torch.save(dual_policy.state_dict(), os.path.join(checkpoint_dir, f"dual_policy_{env_steps}.pth"))
            if save_full_state:
                full_state_path = os.path.join(checkpoint_dir, f"{full_state_prefix}_{env_steps}.pth")
                _save_full_state(
                    full_state_path,
                    env_steps=env_steps,
                    world_model=world_model,
                    agent=agent,
                    gp_critic=gp_critic,
                    gd_critic=gd_critic,
                    dual_policy=dual_policy,
                    replay_buffer=replay_buffer,
                    save_replay_buffer=save_replay_buffer,
                    save_optimizer=save_optimizer,
                )
                print(colorama.Fore.GREEN + f"Saved FDPI reachability Dreamer full state to {full_state_path}" + colorama.Style.RESET_ALL)
            timer.stop("checkpoint", token)

        timer.log(logger, env_steps)

    if offline_episode_writer is not None:
        offline_episode_writer.flush_pending(max_steps)
        print(
            colorama.Fore.CYAN
            + (
                f"Saved {offline_episode_writer.num_saved_episodes} offline episodes "
                f"({offline_episode_writer.num_saved_steps} steps) to {offline_episode_writer.output_dir}"
            )
            + colorama.Style.RESET_ALL
        )
