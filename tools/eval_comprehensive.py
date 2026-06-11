from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


simulation_app = None


def _launch_isaac(headless=True):
    global simulation_app
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    import omni.isaac.lab_tasks  # noqa: F401
    import ur3_lite.tasks  # noqa: F401


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, value):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sq_sum += float(np.square(arr).sum())
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))

    def as_dict(self):
        if self.count <= 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.sum / self.count
        var = max(self.sq_sum / self.count - mean * mean, 0.0)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "min": float(self.min),
            "max": float(self.max),
        }


class BinaryStats:
    def __init__(self):
        self.scores = []
        self.labels = []

    def add(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1) > 0.5
        valid = np.isfinite(scores)
        if valid.any():
            self.scores.append(scores[valid])
            self.labels.append(labels[valid])

    def as_dict(self, threshold=0.5):
        if not self.scores:
            return {
                "count": 0,
                "positive_rate": None,
                "auc": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }
        scores = np.concatenate(self.scores)
        labels = np.concatenate(self.labels).astype(bool)
        pred = scores >= float(threshold)
        tp = float((pred & labels).sum())
        fp = float((pred & ~labels).sum())
        fn = float((~pred & labels).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "count": int(scores.size),
            "positive_rate": float(labels.mean()) if labels.size else None,
            "auc": _binary_auc(scores, labels),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "score_mean": float(scores.mean()) if scores.size else None,
            "score_pos_mean": float(scores[labels].mean()) if labels.any() else None,
            "score_neg_mean": float(scores[~labels].mean()) if (~labels).any() else None,
        }


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


def _episode_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _cfg_get(node, name, default=None):
    if node is None:
        return default
    if hasattr(node, name):
        return getattr(node, name)
    if isinstance(node, dict):
        return node.get(name, default)
    return default


def _cfg_int_tuple(node, name, default=()):
    value = _cfg_get(node, name, default)
    if value is None:
        return tuple()
    return tuple(int(v) for v in value)


def _cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: _cfg_to_dict(value) for key, value in node.items()}
    return node


def _set_eval_num_envs(conf, num_envs):
    if num_envs is None:
        return conf
    from yacs.config import CfgNode as CN

    conf.defrost()
    if not hasattr(conf, "Env"):
        conf.Env = CN(new_allowed=True)
    if not hasattr(conf.Env, "MakeKwargs"):
        conf.Env.MakeKwargs = CN(new_allowed=True)
    conf.Env.MakeKwargs.num_envs = int(num_envs)
    conf.JointTrainAgent.NumEnvs = int(num_envs)
    conf.freeze()
    return conf


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


def _resolve_config_path(checkpoint_dir, explicit_config):
    if explicit_config:
        return os.path.abspath(os.path.expanduser(explicit_config))
    local_config = os.path.join(checkpoint_dir, "config.yaml")
    if os.path.isfile(local_config):
        return local_config
    run_info_path = os.path.join(checkpoint_dir, "run_info.json")
    if os.path.isfile(run_info_path):
        with open(run_info_path, "r", encoding="utf-8") as fin:
            run_info = json.load(fin)
        path = run_info.get("config_path")
        if path and os.path.isfile(path):
            return os.path.abspath(os.path.expanduser(path))
    return os.path.join(PROJECT_ROOT, "configs", "reachability_gp.yaml")


def _tensor_to_np(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_torch_vector(value, device, num_envs, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float32
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(num_envs)
    return tensor[:num_envs]


def _sample_policy_action(policy, feat, greedy):
    return policy.sample(feat, greedy=greedy).to(device=feat.device, dtype=feat.dtype)


def _main_distribution(agent, feat):
    from fdpi_reachability_dreamer.cost_utils import dreamer_agent_distribution

    return dreamer_agent_distribution(agent, feat)


def _extract_episode_flags(info, next_obs_dict, done, device, num_envs):
    import torch

    terminal = torch.as_tensor(
        next_obs_dict.get("is_terminal", torch.zeros_like(done, dtype=torch.int32)),
        dtype=torch.bool,
        device=device,
    ).view(num_envs)
    failure = torch.as_tensor(
        next_obs_dict.get("failure", torch.zeros_like(done, dtype=torch.int32)),
        dtype=torch.bool,
        device=device,
    ).view(num_envs)
    if isinstance(info, dict) and "episode_success" in info:
        success = torch.as_tensor(info["episode_success"], dtype=torch.bool, device=device).view(num_envs)
    else:
        success = terminal & ~failure
    if isinstance(info, dict) and "episode_failure" in info:
        failure_flag = torch.as_tensor(info["episode_failure"], dtype=torch.bool, device=device).view(num_envs)
    else:
        failure_flag = terminal & failure
    if isinstance(info, dict) and "episode_timeout" in info:
        timeout = torch.as_tensor(info["episode_timeout"], dtype=torch.bool, device=device).view(num_envs)
    else:
        timeout = done & ~terminal
    return success, failure_flag, timeout


@dataclass
class RolloutData:
    policy: str
    obs: np.ndarray
    next_obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    is_first: np.ndarray
    continuous_cost: np.ndarray
    binary_cost: np.ndarray
    extreme_cost: np.ndarray
    cost_force: np.ndarray
    summary: dict
    step_stats: dict


def collect_rollout(args, conf, modules, policy_name, policy_index, vec_env=None):
    import torch
    from tqdm import tqdm

    from fdpi_reachability_dreamer.trainer_base import _is_first, _policy_obs, _reset_after_step
    from fdpi_reachability_dreamer.cost_utils import extract_continuous_cost
    from fdpi_reachability_dreamer.train import build_env

    world_model, agent, gp_critic, gd_critic, dual_policy = modules
    device = args.device
    if args.seed is not None:
        conf.defrost()
        conf.BasicSettings.Seed = int(args.seed) + int(policy_index)
        if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
            conf.Env.MakeKwargs.seed = int(args.seed) + int(policy_index)
        conf.freeze()

    owns_env = vec_env is None
    if vec_env is None:
        vec_env = build_env(args, conf)
    try:
        num_envs = int(vec_env.num_envs)
        fdpi_cfg = conf.FDPIRegimeDreamer
        cost_cfg = fdpi_cfg.ContinuousCost
        risk_cfg = fdpi_cfg.RiskCritic
        wm_sampling_cfg = fdpi_cfg.WorldModelSampling
        pf = float(_cfg_get(risk_cfg, "Pf", 0.30))
        cg = float(_cfg_get(risk_cfg, "Cg", 0.15))
        high_cost_threshold = float(_cfg_get(wm_sampling_cfg, "HighCostThreshold", 0.1))
        cost_source = str(_cfg_get(cost_cfg, "CostSource", "bottom"))
        bottom_channels = _cfg_int_tuple(cost_cfg, "BottomForceChannels", [2, 5])
        wall_channels = _cfg_int_tuple(cost_cfg, "WallForceChannels", [1, 4])
        explicit_channels = _cfg_int_tuple(cost_cfg, "CostForceChannels", ())

        state = world_model.initial(num_envs)
        current_obs_dict = vec_env.reset()
        current_obs = _policy_obs(current_obs_dict).to(device)
        is_first = _is_first(current_obs_dict, num_envs, device)

        step_stats = defaultdict(RunningStats)
        episode_returns = []
        episode_lengths = []
        episode_costs = []
        episode_cost_means = []
        episode_binary_cost_counts = []
        episode_extreme_cost_counts = []
        episode_force_peaks = []
        episode_successes = 0
        episode_failures = 0
        episode_timeouts = 0
        episodes_completed = 0

        ep_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ep_cost = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ep_binary = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ep_extreme = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ep_force_peak = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ep_len = torch.zeros(num_envs, dtype=torch.float32, device=device)

        obs_steps = []
        next_obs_steps = []
        action_steps = []
        reward_steps = []
        done_steps = []
        first_steps = []
        cost_steps = []
        binary_steps = []
        extreme_steps = []
        cost_force_steps = []

        total_iters = max(int(math.ceil(float(args.eval_steps) / float(num_envs))), 1)
        if args.eval_episodes is not None:
            total_iters = max(total_iters, 1)
        if args.max_iters is not None:
            total_iters = min(total_iters, int(args.max_iters))
        if args.smoke:
            total_iters = min(total_iters, 8)

        progress = tqdm(total=total_iters, desc=f"Rollout {policy_name}")
        iter_idx = 0
        while True:
            if args.eval_episodes is None and iter_idx >= total_iters:
                break
            if args.eval_episodes is not None and episodes_completed >= int(args.eval_episodes):
                break
            if args.max_iters is not None and iter_idx >= int(args.max_iters):
                break

            with torch.no_grad():
                world_model.eval()
                agent.eval()
                dual_policy.eval()
                gp_critic.eval()
                gd_critic.eval()
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                if policy_name == "main":
                    action = _sample_policy_action(agent, feat, greedy=args.greedy)
                elif policy_name == "dual":
                    action = _sample_policy_action(dual_policy, feat, greedy=args.greedy)
                elif policy_name == "random":
                    sampled = vec_env.action_space.sample()
                    action = torch.as_tensor(sampled, dtype=torch.float32, device=device).reshape(num_envs, -1)
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                gp = gp_critic.risk_no_grad(feat, action, clamp=True).reshape(num_envs)
                gd = gd_critic.risk_no_grad(feat, action, clamp=True).reshape(num_envs)
                main_dist = _main_distribution(agent, feat)
                dual_dist = dual_policy.distribution(feat)
                main_logp = main_dist.log_prob(action).reshape(num_envs)
                dual_logp = dual_dist.log_prob(action).reshape(num_envs)
                main_entropy = main_dist.entropy().reshape(num_envs)
                dual_entropy = dual_dist.entropy().reshape(num_envs)
                state = world_model.update_inference_state(state, action)
                env_action = action.detach().cpu().numpy()

            obs_before_np = current_obs.detach().cpu().numpy()
            is_first_np = is_first.detach().cpu().numpy().reshape(num_envs)
            next_obs_dict, reward, done, info = vec_env.step(env_action)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=device).view(num_envs)
            done = torch.as_tensor(done, dtype=torch.bool, device=device).view(num_envs)
            next_obs = _policy_obs(next_obs_dict).to(device)

            cost_parts = extract_continuous_cost(
                info,
                next_obs_dict,
                num_envs=num_envs,
                device=device,
                force_threshold=float(_cfg_get(cost_cfg, "ForceThreshold", 0.1)),
                low_force_scale=float(_cfg_get(cost_cfg, "LowForceScale", 0.05)),
                cost_force_max=float(_cfg_get(cost_cfg, "CostForceMax", 15.0)),
                force_scale=float(_cfg_get(cost_cfg, "ForceScale", 5.0)),
                extreme_force_threshold=float(_cfg_get(cost_cfg, "ExtremeForceThreshold", 5.0)),
                clip_cost=bool(_cfg_get(cost_cfg, "ClipCost", True)),
                cost_min=float(_cfg_get(cost_cfg, "CostMin", 0.0)),
                cost_max=float(_cfg_get(cost_cfg, "CostMax", 1.0)),
                force_key=str(getattr(conf.ForceHead, "Key", "")),
                cost_source=cost_source,
                bottom_force_channels=bottom_channels,
                wall_force_channels=wall_channels,
                cost_force_channels=explicit_channels if explicit_channels else None,
            )
            continuous_cost = cost_parts["continuous_cost"].view(num_envs)
            binary_cost = cost_parts["binary_cost"].view(num_envs)
            extreme_cost = cost_parts["extreme_cost"].view(num_envs)
            cost_force = cost_parts.get("cost_force", cost_parts.get("bottom_force")).view(num_envs)

            obs_steps.append(obs_before_np.astype(np.float32, copy=True))
            next_obs_steps.append(next_obs.detach().cpu().numpy().astype(np.float32, copy=True))
            action_steps.append(action.detach().cpu().numpy().astype(np.float32, copy=True))
            reward_steps.append(reward.detach().cpu().numpy().astype(np.float32, copy=True))
            done_steps.append(done.detach().cpu().numpy().astype(np.float32, copy=True))
            first_steps.append(is_first_np.astype(np.float32, copy=True))
            cost_steps.append(continuous_cost.detach().cpu().numpy().astype(np.float32, copy=True))
            binary_steps.append(binary_cost.detach().cpu().numpy().astype(np.float32, copy=True))
            extreme_steps.append(extreme_cost.detach().cpu().numpy().astype(np.float32, copy=True))
            cost_force_steps.append(cost_force.detach().cpu().numpy().astype(np.float32, copy=True))

            step_stats["reward"].add(reward.detach().cpu().numpy())
            step_stats["continuous_cost"].add(continuous_cost.detach().cpu().numpy())
            step_stats["binary_cost"].add(binary_cost.detach().cpu().numpy())
            step_stats["high_cost"].add((continuous_cost > high_cost_threshold).float().detach().cpu().numpy())
            step_stats["extreme_cost"].add(extreme_cost.detach().cpu().numpy())
            step_stats["cost_force"].add(cost_force.detach().cpu().numpy())
            step_stats["gp"].add(gp.detach().cpu().numpy())
            step_stats["gd"].add(gd.detach().cpu().numpy())
            step_stats["gp_feasible"].add((gp < (pf - cg)).float().detach().cpu().numpy())
            step_stats["gp_critical"].add(((gp >= (pf - cg)) & (gp < pf)).float().detach().cpu().numpy())
            step_stats["gp_infeasible"].add((gp >= pf).float().detach().cpu().numpy())
            step_stats["main_logp"].add(main_logp.detach().cpu().numpy())
            step_stats["dual_logp"].add(dual_logp.detach().cpu().numpy())
            step_stats["logp_dual_minus_main"].add((dual_logp - main_logp).detach().cpu().numpy())
            step_stats["main_entropy"].add(main_entropy.detach().cpu().numpy())
            step_stats["dual_entropy"].add(dual_entropy.detach().cpu().numpy())
            step_stats["action_abs_mean"].add(action.abs().mean(dim=-1).detach().cpu().numpy())
            step_stats["action_l2"].add(action.float().norm(dim=-1).detach().cpu().numpy())

            success, failure, timeout = _extract_episode_flags(info, next_obs_dict, done, device, num_envs)
            ep_return += reward
            ep_cost += continuous_cost
            ep_binary += binary_cost
            ep_extreme += extreme_cost
            ep_force_peak = torch.maximum(ep_force_peak, cost_force)
            ep_len += 1.0

            if done.any():
                done_indices = torch.nonzero(done, as_tuple=False).flatten()
                episodes_completed += int(done_indices.numel())
                episode_successes += int(success[done_indices].sum().item())
                episode_failures += int(failure[done_indices].sum().item())
                episode_timeouts += int(timeout[done_indices].sum().item())
                for env_idx in done_indices.tolist():
                    length = max(float(ep_len[env_idx].item()), 1.0)
                    episode_returns.append(float(ep_return[env_idx].item()))
                    episode_lengths.append(length)
                    episode_costs.append(float(ep_cost[env_idx].item()))
                    episode_cost_means.append(float(ep_cost[env_idx].item()) / length)
                    episode_binary_cost_counts.append(float(ep_binary[env_idx].item()))
                    episode_extreme_cost_counts.append(float(ep_extreme[env_idx].item()))
                    episode_force_peaks.append(float(ep_force_peak[env_idx].item()))
                    ep_return[env_idx] = 0.0
                    ep_cost[env_idx] = 0.0
                    ep_binary[env_idx] = 0.0
                    ep_extreme[env_idx] = 0.0
                    ep_force_peak[env_idx] = 0.0
                    ep_len[env_idx] = 0.0

            current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, device)
            iter_idx += 1
            progress.update(1)
        progress.close()

        step_dict = {key: stat.as_dict() for key, stat in step_stats.items()}
        binary_count_arr = np.asarray(episode_binary_cost_counts, dtype=np.float64)
        extreme_count_arr = np.asarray(episode_extreme_cost_counts, dtype=np.float64)
        summary = {
            "policy": policy_name,
            "num_envs": int(num_envs),
            "env_steps": int(iter_idx * num_envs),
            "iterations": int(iter_idx),
            "episodes_completed": int(episodes_completed),
            "success_rate": float(episode_successes / max(episodes_completed, 1)),
            "failure_rate": float(episode_failures / max(episodes_completed, 1)),
            "timeout_rate": float(episode_timeouts / max(episodes_completed, 1)),
            "episode_return": _episode_summary(episode_returns),
            "episode_length": _episode_summary(episode_lengths),
            "episode_cost": _episode_summary(episode_costs),
            "episode_cost_mean": _episode_summary(episode_cost_means),
            "episode_binary_cost_count": _episode_summary(episode_binary_cost_counts),
            "episode_cost_trigger_rate": (
                float((binary_count_arr > 0.0).mean()) if binary_count_arr.size else None
            ),
            "episode_cost_free_rate": (
                float((binary_count_arr <= 0.0).mean()) if binary_count_arr.size else None
            ),
            "episode_extreme_cost_count": _episode_summary(episode_extreme_cost_counts),
            "episode_extreme_cost_trigger_rate": (
                float((extreme_count_arr > 0.0).mean()) if extreme_count_arr.size else None
            ),
            "episode_cost_force_peak": _episode_summary(episode_force_peaks),
            "continuous_cost_mean": step_dict["continuous_cost"]["mean"],
            "cost_positive_rate": step_dict["binary_cost"]["mean"],
            "high_cost_rate": step_dict["high_cost"]["mean"],
            "extreme_cost_rate": step_dict["extreme_cost"]["mean"],
            "cost_force_mean": step_dict["cost_force"]["mean"],
            "cost_force_max": step_dict["cost_force"]["max"],
            "gp_mean": step_dict["gp"]["mean"],
            "gd_mean": step_dict["gd"]["mean"],
            "gp_feasible_ratio": step_dict["gp_feasible"]["mean"],
            "gp_critical_ratio": step_dict["gp_critical"]["mean"],
            "gp_infeasible_ratio": step_dict["gp_infeasible"]["mean"],
            "logp_dual_minus_main_mean": step_dict["logp_dual_minus_main"]["mean"],
            "main_entropy_mean": step_dict["main_entropy"]["mean"],
            "dual_entropy_mean": step_dict["dual_entropy"]["mean"],
            "action_l2_mean": step_dict["action_l2"]["mean"],
        }
        return RolloutData(
            policy=policy_name,
            obs=np.stack(obs_steps, axis=0),
            next_obs=np.stack(next_obs_steps, axis=0),
            action=np.stack(action_steps, axis=0),
            reward=np.stack(reward_steps, axis=0),
            done=np.stack(done_steps, axis=0),
            is_first=np.stack(first_steps, axis=0),
            continuous_cost=np.stack(cost_steps, axis=0),
            binary_cost=np.stack(binary_steps, axis=0),
            extreme_cost=np.stack(extreme_steps, axis=0),
            cost_force=np.stack(cost_force_steps, axis=0),
            summary=summary,
            step_stats=step_dict,
        )
    finally:
        if owns_env:
            vec_env.close()


def _valid_model_windows(rollout: RolloutData, warmup, horizon):
    done = rollout.done > 0.5
    total, num_envs = done.shape
    max_start = total - int(warmup) - int(horizon)
    if max_start < 0:
        return []
    valid = []
    span = int(warmup) + int(horizon)
    for t in range(max_start + 1):
        window_done = done[t : t + span]
        env_valid = ~window_done.any(axis=0)
        for env_idx in np.nonzero(env_valid)[0].tolist():
            valid.append((t, env_idx))
    return valid


def _sample_windows(valid, max_windows, seed):
    if not valid:
        return []
    max_windows = int(max_windows)
    if max_windows <= 0 or len(valid) <= max_windows:
        return valid
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(valid), size=max_windows, replace=False)
    return [valid[int(i)] for i in ids]


def _batch_from_windows(rollout: RolloutData, windows, warmup, horizon, batch_start, batch_size, device):
    import torch

    selected = windows[batch_start : batch_start + batch_size]
    bsz = len(selected)
    obs_warm = np.empty((bsz, warmup + 1, rollout.obs.shape[-1]), dtype=np.float32)
    act_all = np.empty((bsz, warmup + horizon, rollout.action.shape[-1]), dtype=np.float32)
    first_warm = np.zeros((bsz, warmup + 1, 1), dtype=np.float32)
    targets = {
        "obs": np.empty((bsz, horizon, rollout.obs.shape[-1]), dtype=np.float32),
        "reward": np.empty((bsz, horizon, 1), dtype=np.float32),
        "done": np.empty((bsz, horizon, 1), dtype=np.float32),
        "cost": np.empty((bsz, horizon, 1), dtype=np.float32),
        "binary_cost": np.empty((bsz, horizon, 1), dtype=np.float32),
        "extreme_cost": np.empty((bsz, horizon, 1), dtype=np.float32),
    }
    for row, (t, env_idx) in enumerate(selected):
        obs_warm[row, :warmup] = rollout.obs[t : t + warmup, env_idx]
        obs_warm[row, warmup] = rollout.obs[t + warmup, env_idx]
        act_all[row] = rollout.action[t : t + warmup + horizon, env_idx]
        first_warm[row, :, 0] = rollout.is_first[t : t + warmup + 1, env_idx]
        start = t + warmup
        stop = start + horizon
        targets["obs"][row] = rollout.next_obs[start:stop, env_idx]
        targets["reward"][row, :, 0] = rollout.reward[start:stop, env_idx]
        targets["done"][row, :, 0] = rollout.done[start:stop, env_idx]
        targets["cost"][row, :, 0] = rollout.continuous_cost[start:stop, env_idx]
        targets["binary_cost"][row, :, 0] = rollout.binary_cost[start:stop, env_idx]
        targets["extreme_cost"][row, :, 0] = rollout.extreme_cost[start:stop, env_idx]
    return (
        torch.as_tensor(obs_warm, dtype=torch.float32, device=device),
        torch.as_tensor(act_all, dtype=torch.float32, device=device),
        torch.as_tensor(first_warm, dtype=torch.float32, device=device),
        {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in targets.items()},
    )


def evaluate_world_model(args, world_model, rollout: RolloutData, policy_index=0):
    import torch

    world_model.eval()
    warmup = int(args.model_warmup)
    horizon = int(args.model_horizon)
    valid = _valid_model_windows(rollout, warmup, horizon)
    valid = _sample_windows(valid, args.model_eval_windows, int(args.seed or 0) + 1000 + policy_index)
    if not valid:
        return {"policy": rollout.policy, "num_windows": 0, "error": "no valid non-terminal windows"}

    obs_dim = int(rollout.obs.shape[-1])
    horizon_stats = {
        h: {
            "obs_abs": RunningStats(),
            "obs_sq": RunningStats(),
            "reward_abs": RunningStats(),
            "reward_sq": RunningStats(),
            "cost_abs": RunningStats(),
            "cost_sq": RunningStats(),
            "done": BinaryStats(),
            "cost_event": BinaryStats(),
            "extreme_event": BinaryStats(),
        }
        for h in range(1, horizon + 1)
    }
    per_dim_abs_h1 = np.zeros(obs_dim, dtype=np.float64)
    per_dim_sq_h1 = np.zeros(obs_dim, dtype=np.float64)
    per_dim_count_h1 = np.zeros(obs_dim, dtype=np.float64)
    per_dim_abs_hn = np.zeros(obs_dim, dtype=np.float64)
    per_dim_sq_hn = np.zeros(obs_dim, dtype=np.float64)
    per_dim_count_hn = np.zeros(obs_dim, dtype=np.float64)
    scatter_true = []
    scatter_pred = []

    batch_size = int(args.model_batch_size)
    device = args.device
    with torch.no_grad():
        for batch_start in range(0, len(valid), batch_size):
            obs_warm, act_all, first_warm, targets = _batch_from_windows(
                rollout,
                valid,
                warmup,
                horizon,
                batch_start,
                batch_size,
                device,
            )
            bsz = int(obs_warm.shape[0])
            state = world_model.initial(bsz)
            # Training-style warmup: observe real states, advance latent with real actions.
            for t in range(warmup + 1):
                _, state = world_model.get_inference_feat(state, obs_warm[:, t], first_warm[:, t])
                if t < warmup:
                    state = world_model.update_inference_state(state, act_all[:, t])

            pred_obs_steps = []
            pred_reward_steps = []
            pred_done_steps = []
            pred_cost_steps = []
            pred_extreme_steps = []
            for h in range(horizon):
                action = act_all[:, warmup + h]
                state = world_model.dynamic.img_step(state, action)
                stoch = world_model.dynamic.get_flatten_stoch(state)
                feat = world_model.dynamic.get_feat(state)
                pred_obs = world_model.decode_obs(stoch).float() if hasattr(world_model, "decode_obs") else world_model.decoder(stoch).float()
                pred_reward = world_model.twohot_loss.decode(world_model.reward_head(state["deter"])).float()
                pred_done_prob = torch.sigmoid(world_model.done_head(state["deter"])).float()
                if hasattr(world_model, "predict_cost"):
                    pred_cost, pred_extreme_prob, _ = world_model.predict_cost(feat)
                    pred_cost = pred_cost.float()
                    pred_extreme_prob = pred_extreme_prob.float()
                else:
                    pred_cost = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
                    pred_extreme_prob = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
                pred_obs_steps.append(pred_obs)
                pred_reward_steps.append(pred_reward.reshape(bsz, 1))
                pred_done_steps.append(pred_done_prob.reshape(bsz, 1))
                pred_cost_steps.append(pred_cost.reshape(bsz, 1))
                pred_extreme_steps.append(pred_extreme_prob.reshape(bsz, 1))

            pred_obs = torch.stack(pred_obs_steps, dim=1)
            pred_reward = torch.stack(pred_reward_steps, dim=1)
            pred_done = torch.stack(pred_done_steps, dim=1)
            pred_cost = torch.stack(pred_cost_steps, dim=1)
            pred_extreme = torch.stack(pred_extreme_steps, dim=1)

            obs_err = (pred_obs - targets["obs"]).detach().float()
            rew_err = (pred_reward - targets["reward"]).detach().float()
            cost_err = (pred_cost - targets["cost"]).detach().float()
            for h in range(horizon):
                stats = horizon_stats[h + 1]
                stats["obs_abs"].add(obs_err[:, h].abs().cpu().numpy())
                stats["obs_sq"].add(obs_err[:, h].pow(2).cpu().numpy())
                stats["reward_abs"].add(rew_err[:, h].abs().cpu().numpy())
                stats["reward_sq"].add(rew_err[:, h].pow(2).cpu().numpy())
                stats["cost_abs"].add(cost_err[:, h].abs().cpu().numpy())
                stats["cost_sq"].add(cost_err[:, h].pow(2).cpu().numpy())
                stats["done"].add(pred_done[:, h].cpu().numpy(), targets["done"][:, h].cpu().numpy())
                stats["cost_event"].add(pred_cost[:, h].cpu().numpy(), targets["binary_cost"][:, h].cpu().numpy())
                stats["extreme_event"].add(
                    pred_extreme[:, h].cpu().numpy(),
                    targets["extreme_cost"][:, h].cpu().numpy(),
                )

            h1_abs = obs_err[:, 0].abs().cpu().numpy()
            h1_sq = obs_err[:, 0].pow(2).cpu().numpy()
            hn_abs = obs_err[:, -1].abs().cpu().numpy()
            hn_sq = obs_err[:, -1].pow(2).cpu().numpy()
            per_dim_abs_h1 += h1_abs.sum(axis=0)
            per_dim_sq_h1 += h1_sq.sum(axis=0)
            per_dim_count_h1 += h1_abs.shape[0]
            per_dim_abs_hn += hn_abs.sum(axis=0)
            per_dim_sq_hn += hn_sq.sum(axis=0)
            per_dim_count_hn += hn_abs.shape[0]
            if len(scatter_true) < 8:
                scatter_true.append(targets["cost"][:, 0].detach().cpu().numpy().reshape(-1))
                scatter_pred.append(pred_cost[:, 0].detach().cpu().numpy().reshape(-1))

    per_horizon = {}
    for h, stats in horizon_stats.items():
        obs_mse = stats["obs_sq"].sum / max(stats["obs_sq"].count, 1)
        reward_mse = stats["reward_sq"].sum / max(stats["reward_sq"].count, 1)
        cost_mse = stats["cost_sq"].sum / max(stats["cost_sq"].count, 1)
        per_horizon[str(h)] = {
            "obs_mae": stats["obs_abs"].as_dict()["mean"],
            "obs_rmse": float(math.sqrt(obs_mse)),
            "reward_mae": stats["reward_abs"].as_dict()["mean"],
            "reward_rmse": float(math.sqrt(reward_mse)),
            "cost_mae": stats["cost_abs"].as_dict()["mean"],
            "cost_rmse": float(math.sqrt(cost_mse)),
            "done_event": stats["done"].as_dict(threshold=0.5),
            "cost_event": stats["cost_event"].as_dict(threshold=float(args.cost_event_threshold)),
            "extreme_event": stats["extreme_event"].as_dict(threshold=0.5),
        }

    def per_dim_dict(abs_sum, sq_sum, count):
        count = np.maximum(count, 1.0)
        return {
            "mae": (abs_sum / count).astype(float).tolist(),
            "rmse": np.sqrt(sq_sum / count).astype(float).tolist(),
        }

    true_cost = np.concatenate(scatter_true).astype(float).tolist() if scatter_true else []
    pred_cost = np.concatenate(scatter_pred).astype(float).tolist() if scatter_pred else []
    max_scatter = int(args.plot_scatter_points)
    if len(true_cost) > max_scatter > 0:
        true_cost = true_cost[:max_scatter]
        pred_cost = pred_cost[:max_scatter]
    return {
        "policy": rollout.policy,
        "num_windows": int(len(valid)),
        "warmup": int(warmup),
        "horizon": int(horizon),
        "per_horizon": per_horizon,
        "obs_dim_h1": per_dim_dict(per_dim_abs_h1, per_dim_sq_h1, per_dim_count_h1),
        "obs_dim_horizon": per_dim_dict(per_dim_abs_hn, per_dim_sq_hn, per_dim_count_hn),
        "cost_scatter_h1": {"true": true_cost, "pred": pred_cost},
    }


def _plot_results(save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    plot_paths = {}
    policies = result["policies"]
    summaries = result["policy_rollouts"]

    labels = policies
    success = [summaries[p]["summary"].get("success_rate") or 0.0 for p in labels]
    cost_trigger = [summaries[p]["summary"].get("episode_cost_trigger_rate") or 0.0 for p in labels]
    cost_rate = [summaries[p]["summary"].get("cost_positive_rate") or 0.0 for p in labels]
    returns = [summaries[p]["summary"].get("episode_return", {}).get("mean") or 0.0 for p in labels]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    width = 0.26
    axes[0].bar(x - width, success, width, label="success")
    axes[0].bar(x, cost_trigger, width, label="episode cost trigger")
    axes[0].bar(x + width, cost_rate, width, label="step cost positive")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Task and safety rates")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, returns, width=0.55, color="#3c7d87")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Episode return mean")
    axes[1].set_ylabel("return")
    path = os.path.join(save_dir, "policy_task_safety.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["policy_task_safety"] = path

    risk_gp = [summaries[p]["summary"].get("gp_mean") or 0.0 for p in labels]
    risk_gd = [summaries[p]["summary"].get("gd_mean") or 0.0 for p in labels]
    infeasible = [summaries[p]["summary"].get("gp_infeasible_ratio") or 0.0 for p in labels]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x - width, risk_gp, width, label="Gp")
    ax.bar(x, risk_gd, width, label="Gd")
    ax.bar(x + width, infeasible, width, label="Gp infeasible ratio")
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_title("Risk critics by rollout policy")
    ax.legend(fontsize=8)
    path = os.path.join(save_dir, "risk_critics.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["risk_critics"] = path

    wm = result.get("world_model", {})
    if wm:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for policy in labels:
            item = wm.get(policy)
            if not item or not item.get("per_horizon"):
                continue
            horizons = sorted((int(k) for k in item["per_horizon"].keys()))
            axes[0].plot(horizons, [item["per_horizon"][str(h)]["obs_rmse"] for h in horizons], marker="o", label=policy)
            axes[1].plot(horizons, [item["per_horizon"][str(h)]["cost_mae"] for h in horizons], marker="o", label=policy)
            axes[2].plot(horizons, [item["per_horizon"][str(h)]["reward_rmse"] for h in horizons], marker="o", label=policy)
        axes[0].set_title("Observation RMSE")
        axes[1].set_title("Cost MAE")
        axes[2].set_title("Reward RMSE")
        for ax in axes:
            ax.set_xlabel("open-loop horizon")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        path = os.path.join(save_dir, "world_model_horizon_errors.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["world_model_horizon_errors"] = path

        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        for policy in labels:
            item = wm.get(policy)
            if not item or not item.get("obs_dim_h1"):
                continue
            ax.plot(item["obs_dim_h1"]["mae"], label=f"{policy} h=1", linewidth=1.5)
        ax.set_title("Per-observation-dimension one-step MAE")
        ax.set_xlabel("observation dimension")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        path = os.path.join(save_dir, "obs_dim_one_step_mae.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["obs_dim_one_step_mae"] = path

        fig, axes = plt.subplots(1, max(len(labels), 1), figsize=(4 * max(len(labels), 1), 4), constrained_layout=True)
        if len(labels) == 1:
            axes = [axes]
        for ax, policy in zip(axes, labels):
            item = wm.get(policy)
            if not item:
                ax.set_axis_off()
                continue
            scatter = item.get("cost_scatter_h1", {})
            true = np.asarray(scatter.get("true", []), dtype=float)
            pred = np.asarray(scatter.get("pred", []), dtype=float)
            if true.size:
                ax.scatter(true, pred, s=8, alpha=0.4)
            ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{policy} cost h=1")
            ax.set_xlabel("true cost")
            ax.set_ylabel("pred cost")
        path = os.path.join(save_dir, "cost_prediction_scatter_h1.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["cost_prediction_scatter_h1"] = path
    return plot_paths


def _save_report(save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    step = result.get("checkpoint_step") or "unknown"
    json_path = os.path.join(save_dir, f"dfd_v5_comprehensive_{step}_{stamp}.json")
    md_path = os.path.join(save_dir, f"dfd_v5_comprehensive_{step}_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

    lines = [
        "# DFD v5 Comprehensive Evaluation",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- config: `{result['config_path']}`",
        f"- env: `{result['env_name']}`",
        f"- num_envs: `{result['num_envs']}`",
        f"- policies: `{', '.join(result['policies'])}`",
        f"- cost_source: `{result['cost_config'].get('CostSource')}`",
        "",
        "## Policy Rollouts",
        "",
        "| policy | episodes | success | return_mean | cost_trigger | step_cost_rate | gp_mean | gd_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in result["policies"]:
        summary = result["policy_rollouts"][policy]["summary"]
        ret = summary.get("episode_return", {}).get("mean")
        lines.append(
            "| {policy} | {eps} | {success:.4f} | {ret} | {trigger} | {cost_rate} | {gp} | {gd} |".format(
                policy=policy,
                eps=summary.get("episodes_completed"),
                success=float(summary.get("success_rate") or 0.0),
                ret="None" if ret is None else f"{ret:.4f}",
                trigger=summary.get("episode_cost_trigger_rate"),
                cost_rate=summary.get("cost_positive_rate"),
                gp=summary.get("gp_mean"),
                gd=summary.get("gd_mean"),
            )
        )
    lines += ["", "## World Model", ""]
    for policy in result["policies"]:
        item = result.get("world_model", {}).get(policy, {})
        if not item or not item.get("per_horizon"):
            lines.append(f"- {policy}: no valid model-eval windows")
            continue
        h1 = item["per_horizon"].get("1", {})
        hn = item["per_horizon"].get(str(item.get("horizon")), {})
        lines.append(
            "- {policy}: windows={windows}, h1_obs_rmse={h1_obs}, h1_cost_mae={h1_cost}, "
            "h{h}_obs_rmse={hn_obs}, h{h}_cost_mae={hn_cost}, h1_cost_auc={auc}".format(
                policy=policy,
                windows=item.get("num_windows"),
                h=item.get("horizon"),
                h1_obs=h1.get("obs_rmse"),
                h1_cost=h1.get("cost_mae"),
                hn_obs=hn.get("obs_rmse"),
                hn_cost=hn.get("cost_mae"),
                auc=h1.get("cost_event", {}).get("auc"),
            )
        )
    lines += ["", "## Plots", ""]
    for name, path in result.get("plots", {}).items():
        if name.endswith("error"):
            continue
        rel = os.path.relpath(path, save_dir)
        lines.append(f"- {name}: `{rel}`")
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
    config_path = _resolve_config_path(checkpoint_dir, args.config_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    conf = load_dfd_v5_config(config_path)
    conf = _set_eval_num_envs(conf, args.num_envs)
    if args.seed is not None:
        conf.defrost()
        conf.BasicSettings.Seed = int(args.seed)
        if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
            conf.Env.MakeKwargs.seed = int(args.seed)
        conf.freeze()
    torch.manual_seed(int(args.seed or conf.BasicSettings.Seed))
    np.random.seed(int(args.seed or conf.BasicSettings.Seed))

    # Build the first env to infer dimensions, then reuse it for the first rollout.
    # Closing an IsaacLab env before creating the next one can tear down enough of
    # the simulation context to make a second env creation flaky in headless runs.
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
    world_model.eval()
    agent.eval()
    gp_critic.eval()
    gd_critic.eval()
    dual_policy.eval()
    modules = (world_model, agent, gp_critic, gd_critic, dual_policy)

    policies = [p.strip() for p in str(args.policies).split(",") if p.strip()]
    if args.smoke:
        policies = policies[:1]
    rollout_results = {}
    model_results = {}
    for idx, policy in enumerate(policies):
        rollout_env = first_env if idx == 0 else None
        rollout = collect_rollout(args, conf.clone(), modules, policy, idx, vec_env=rollout_env)
        if idx == 0:
            first_env = None
        rollout_results[policy] = {
            "summary": rollout.summary,
            "step_stats": rollout.step_stats,
        }
        if args.evaluate_world_model:
            model_results[policy] = evaluate_world_model(args, world_model, rollout, idx)

    save_root = os.path.abspath(os.path.expanduser(args.save_dir))
    leaf = f"dfd_v5_{checkpoint_step or 'unknown'}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(save_root, leaf)
    os.makedirs(save_dir, exist_ok=True)
    cost_config = _cfg_to_dict(conf.FDPIRegimeDreamer.ContinuousCost)
    result = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "config_path": config_path,
        "env_name": args.env_name,
        "device": args.device,
        "seed": int(args.seed or conf.BasicSettings.Seed),
        "num_envs": int(args.num_envs),
        "greedy": bool(args.greedy),
        "policies": policies,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "cost_config": cost_config,
        "policy_rollouts": rollout_results,
        "world_model": model_results,
        "config": _cfg_to_dict(conf) if args.save_config else None,
    }
    result["plots"] = _plot_results(save_dir, result)
    json_path, md_path = _save_report(save_dir, result)

    print("\nDFD v5 comprehensive evaluation")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config: {config_path}")
    print(f"  save_dir: {save_dir}")
    for policy in policies:
        summary = rollout_results[policy]["summary"]
        print(
            "  {policy}: episodes={episodes}, success={success:.4f}, "
            "return_mean={ret}, cost_trigger={trigger}, step_cost={cost}, gp={gp}, gd={gd}".format(
                policy=policy,
                episodes=summary.get("episodes_completed"),
                success=float(summary.get("success_rate") or 0.0),
                ret=summary.get("episode_return", {}).get("mean"),
                trigger=summary.get("episode_cost_trigger_rate"),
                cost=summary.get("cost_positive_rate"),
                gp=summary.get("gp_mean"),
                gd=summary.get("gd_mean"),
            )
        )
        wm_item = model_results.get(policy)
        if wm_item and wm_item.get("per_horizon"):
            h1 = wm_item["per_horizon"]["1"]
            hn = wm_item["per_horizon"][str(wm_item["horizon"])]
            print(
                f"    wm: windows={wm_item['num_windows']}, "
                f"h1_obs_rmse={h1['obs_rmse']:.6f}, h1_cost_mae={h1['cost_mae']:.6f}, "
                f"h{wm_item['horizon']}_obs_rmse={hn['obs_rmse']:.6f}, "
                f"h{wm_item['horizon']}_cost_mae={hn['cost_mae']:.6f}"
            )
    print(f"  json: {json_path}")
    print(f"  md: {md_path}")
    return result


def main():
    warnings.filterwarnings("ignore")
    default_ckpt_dir = os.path.join(
        PROJECT_ROOT,
        "ckpt",
        "dfd-v5-task-first-from-750k",
        "20260603_121444",
    )
    parser = argparse.ArgumentParser(description="Comprehensive DFD v5 policy and world-model evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("-config_path", type=str, default=None)
    parser.add_argument("-env_name", type=str, default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=49152)
    parser.add_argument("--eval_episodes", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--policies", type=str, default="main,dual,random")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--evaluate_world_model", action="store_true", default=True)
    parser.add_argument("--skip_world_model", action="store_false", dest="evaluate_world_model")
    parser.add_argument("--model_warmup", type=int, default=16)
    parser.add_argument("--model_horizon", type=int, default=15)
    parser.add_argument("--model_eval_windows", type=int, default=4096)
    parser.add_argument("--model_batch_size", type=int, default=256)
    parser.add_argument("--cost_event_threshold", type=float, default=0.05)
    parser.add_argument("--plot_scatter_points", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "eval_results", "dfd_v5_comprehensive"))
    parser.add_argument("--save_config", action="store_true")
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
