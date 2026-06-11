from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fdpi_reachability_dreamer import train as train_v5


class Timer:
    def __init__(self, torch_module, device: str):
        self.torch = torch_module
        self.device = device

    def sync(self):
        if "cuda" in str(self.device) and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def now(self):
        self.sync()
        return time.perf_counter()

    def elapsed(self, start):
        self.sync()
        return time.perf_counter() - start


def _cfg_int(node, name, default):
    return int(train_v5.cfg_get(node, name, default))


def _cfg_float(node, name, default):
    return float(train_v5.cfg_get(node, name, default))


def _cfg_bool(node, name, default=False):
    return bool(train_v5.cfg_get(node, name, default))


def _cfg_int_tuple(node, name, default=()):
    value = train_v5.cfg_get(node, name, default)
    if value is None:
        return tuple()
    return tuple(int(v) for v in value)


def _mean(values):
    return float(sum(values) / max(len(values), 1))


def _sum(values):
    return float(sum(values))


def _finite(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _hardware_summary():
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return output
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"nvidia-smi unavailable: {exc}"


def _set_config_overrides(conf, args):
    conf.defrost()
    conf.JointTrainAgent.NumEnvs = int(args.num_envs)
    conf.JointTrainAgent.BatchSize = int(args.batch_size)
    conf.JointTrainAgent.BatchLength = int(args.batch_length)
    conf.JointTrainAgent.ImagineBatchSize = int(args.imagine_batch_size)
    conf.JointTrainAgent.ImagineContext = int(args.imagine_context)
    conf.JointTrainAgent.ImagineHorizon = int(args.imagine_horizon)
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.SaveEverySteps = 10**12
    if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
        conf.Env.MakeKwargs.num_envs = int(args.num_envs)
    if args.disable_force_head:
        conf.ForceHead.Enable = False
    if args.no_obs_normalizer:
        conf.Models.WorldModel.ObsNormalizer.Enable = False
    conf.freeze()


def _random_action(vec_env, torch, num_envs, action_dim, device):
    space = vec_env.single_action_space
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        action = torch.empty(num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
    else:
        low_t = torch.as_tensor(low, dtype=torch.float32, device=device).reshape(1, -1)
        high_t = torch.as_tensor(high, dtype=torch.float32, device=device).reshape(1, -1)
        low_t = torch.where(torch.isfinite(low_t), low_t, torch.full_like(low_t, -1.0))
        high_t = torch.where(torch.isfinite(high_t), high_t, torch.full_like(high_t, 1.0))
        action = low_t + torch.rand(num_envs, action_dim, device=device) * (high_t - low_t)
    return action.clamp(-1.0, 1.0)


def _append_transition(
    *,
    replay_buffer,
    current_obs_dict,
    current_obs,
    action,
    reward,
    done,
    is_first,
    next_obs_dict,
    info,
    source,
    cost_cfg,
    device,
):
    from fdpi_reachability_dreamer.trainer_base import _extract_force_obs
    from fdpi_reachability_dreamer.cost_utils import extract_continuous_cost

    num_envs = replay_buffer.num_envs
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
        cost_source=str(train_v5.cfg_get(cost_cfg, "CostSource", "bottom")),
        bottom_force_channels=_cfg_int_tuple(cost_cfg, "BottomForceChannels", [2, 5]),
        wall_force_channels=_cfg_int_tuple(cost_cfg, "WallForceChannels", [1, 4]),
        cost_force_channels=_cfg_int_tuple(cost_cfg, "CostForceChannels", ()),
    )
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
        continuous_cost=cost_parts["continuous_cost"],
        binary_cost=cost_parts["binary_cost"],
        extreme_cost=cost_parts["extreme_cost"],
        bottom_force=cost_parts["bottom_force"],
        force_excess=cost_parts["force_excess"],
        source=source,
    )
    return cost_parts


def _build_everything(args, conf):
    train_args = SimpleNamespace(env_name=args.env_name, device=args.device, seed=args.seed)
    vec_env = train_v5.build_env(train_args, conf)
    obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
    action_dim = int(vec_env.single_action_space.shape[0])
    act = getattr(train_v5.nn, conf.Models.Act)
    world_model = train_v5.build_world_model(conf, obs_dim, action_dim, act, args.device)
    agent = train_v5.build_agent(conf, action_dim, act, args.device)
    gp_critic = train_v5.build_gp_critic(conf, action_dim, act, args.device)
    gd_critic = train_v5.build_gd_critic(conf, action_dim, act, args.device)
    dual_policy = train_v5.build_dual_policy(conf, action_dim, act, args.device)
    if bool(train_v5.cfg_get(conf.FDPIRegimeDreamer.DualPolicy, "InitFromMainActor", True)):
        dual_policy.initialize_from_main_actor(agent)
    replay_buffer = train_v5.DFDV4ReplayBuffer(
        obs_dim,
        action_dim,
        vec_env.num_envs,
        conf.JointTrainAgent.BufferMaxLength,
        0,
        args.device,
        include_force=bool(conf.ForceHead.Enable),
        force_dim=1,
        force_key=conf.ForceHead.Key,
    )
    return vec_env, replay_buffer, world_model, agent, gp_critic, gd_critic, dual_policy, obs_dim, action_dim


def _prefill_replay(args, conf, vec_env, replay_buffer, action_dim, timer):
    torch = train_v5.torch
    from fdpi_reachability_dreamer.trainer_base import _is_first, _policy_obs, _reset_after_step
    from fdpi_reachability_dreamer.cost_utils import SOURCE_RANDOM

    cost_cfg = conf.FDPIRegimeDreamer.ContinuousCost
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(args.device)
    is_first = _is_first(current_obs_dict, vec_env.num_envs, args.device)
    source = torch.full((vec_env.num_envs, 1), SOURCE_RANDOM, dtype=torch.int64, device=args.device)
    row_target = max(int(args.prefill_rows), int(args.batch_length) + int(args.prefill_margin_rows))
    timings = defaultdict(list)
    reward_sum = 0.0
    cost_sum = 0.0
    cost_rate_sum = 0.0

    while replay_buffer.length + 1 < row_target:
        start = timer.now()
        action = _random_action(vec_env, torch, vec_env.num_envs, action_dim, args.device)
        timings["random_action"].append(timer.elapsed(start))

        start = timer.now()
        next_obs_dict, reward, done, info = vec_env.step(action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=args.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=args.device)
        timings["env_step"].append(timer.elapsed(start))

        start = timer.now()
        cost_parts = _append_transition(
            replay_buffer=replay_buffer,
            current_obs_dict=current_obs_dict,
            current_obs=current_obs,
            action=action,
            reward=reward,
            done=done,
            is_first=is_first,
            next_obs_dict=next_obs_dict,
            info=info,
            source=source,
            cost_cfg=cost_cfg,
            device=args.device,
        )
        timings["append"].append(timer.elapsed(start))

        reward_sum += float(reward.detach().float().mean().item())
        cost_sum += float(cost_parts["continuous_cost"].detach().float().mean().item())
        cost_rate_sum += float(cost_parts["binary_cost"].detach().float().mean().item())

        start = timer.now()
        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args.device)
        timings["reset_after_step"].append(timer.elapsed(start))

    rows = replay_buffer.length + 1
    steps = rows * vec_env.num_envs
    return {
        "rows": int(rows),
        "env_steps": int(steps),
        "seconds": {key: _sum(value) for key, value in timings.items()},
        "mean_seconds": {key: _mean(value) for key, value in timings.items()},
        "env_steps_per_sec": float(steps / max(sum(_sum(v) for v in timings.values()), 1.0e-9)),
        "mean_reward": reward_sum / max(rows, 1),
        "mean_continuous_cost": cost_sum / max(rows, 1),
        "mean_binary_cost_rate": cost_rate_sum / max(rows, 1),
    }


def _measure_policy_rollout(args, conf, vec_env, replay_buffer, world_model, agent, gp_critic, dual_policy, action_dim, timer):
    torch = train_v5.torch
    from fdpi_reachability_dreamer.trainer_base import _is_first, _policy_obs, _reset_after_step
    from fdpi_reachability_dreamer.trainer import _sample_policy_action

    cost_cfg = conf.FDPIRegimeDreamer.ContinuousCost
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(args.device)
    is_first = _is_first(current_obs_dict, vec_env.num_envs, args.device)
    state = world_model.initial(vec_env.num_envs)
    timings = defaultdict(list)
    reward_sum = 0.0
    cost_sum = 0.0
    cost_rate_sum = 0.0

    for _ in range(max(int(args.rollout_iters), 0)):
        start = timer.now()
        with torch.no_grad():
            world_model.eval()
            agent.eval()
            feat, state = world_model.get_inference_feat(state, current_obs, is_first)
            env_action, action, source, state, _ = _sample_policy_action(
                feat=feat,
                agent=agent,
                gp_critic=gp_critic,
                dual_policy=dual_policy,
                world_model=world_model,
                state=state,
                use_dual_sampling=float(args.dual_rollout_ratio) > 0.0,
                dual_ratio=float(args.dual_rollout_ratio),
                num_envs=vec_env.num_envs,
                device=args.device,
            )
        timings["policy_action"].append(timer.elapsed(start))

        start = timer.now()
        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=args.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=args.device)
        timings["env_step"].append(timer.elapsed(start))

        start = timer.now()
        cost_parts = _append_transition(
            replay_buffer=replay_buffer,
            current_obs_dict=current_obs_dict,
            current_obs=current_obs,
            action=action,
            reward=reward,
            done=done,
            is_first=is_first,
            next_obs_dict=next_obs_dict,
            info=info,
            source=source,
            cost_cfg=cost_cfg,
            device=args.device,
        )
        timings["append"].append(timer.elapsed(start))
        reward_sum += float(reward.detach().float().mean().item())
        cost_sum += float(cost_parts["continuous_cost"].detach().float().mean().item())
        cost_rate_sum += float(cost_parts["binary_cost"].detach().float().mean().item())

        start = timer.now()
        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args.device)
        timings["reset_after_step"].append(timer.elapsed(start))

    total_seconds = sum(_sum(v) for v in timings.values())
    total_steps = max(int(args.rollout_iters), 0) * vec_env.num_envs
    return {
        "iters": int(args.rollout_iters),
        "env_steps": int(total_steps),
        "seconds": {key: _sum(value) for key, value in timings.items()},
        "mean_seconds": {key: _mean(value) for key, value in timings.items()},
        "total_seconds": float(total_seconds),
        "env_steps_per_sec": float(total_steps / max(total_seconds, 1.0e-9)),
        "mean_reward": reward_sum / max(int(args.rollout_iters), 1),
        "mean_continuous_cost": cost_sum / max(int(args.rollout_iters), 1),
        "mean_binary_cost_rate": cost_rate_sum / max(int(args.rollout_iters), 1),
    }


def _sample_batch(replay_buffer, batch_size, horizon, **kwargs):
    return replay_buffer.sample(batch_size, horizon, return_dict=True, **kwargs)


def _benchmark_phase(
    *,
    args,
    conf,
    replay_buffer,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    phase_step,
    cycles,
    timer,
    record_prefix,
):
    torch = train_v5.torch
    from fdpi_reachability_dreamer.dual_update import update_dual_v4
    from fdpi_reachability_dreamer.trainer import train_agent_step_dfd_v4, train_world_model_step_dfd_v4

    fdpi_cfg = conf.FDPIRegimeDreamer
    wm_sampling_cfg = fdpi_cfg.WorldModelSampling
    gp_cfg = fdpi_cfg.Gp
    gd_cfg = fdpi_cfg.Gd
    dual_update_cfg = fdpi_cfg.DualUpdate
    cost_cfg = fdpi_cfg.ContinuousCost

    high_cost_threshold = _cfg_float(wm_sampling_cfg, "HighCostThreshold", _cfg_float(gp_cfg, "HighCostThreshold", 0.1))
    boundary_low = _cfg_float(wm_sampling_cfg, "BoundaryLow", _cfg_float(gp_cfg, "BoundaryLow", 0.05))
    boundary_high = _cfg_float(wm_sampling_cfg, "BoundaryHigh", _cfg_float(gp_cfg, "BoundaryHigh", 0.4))
    world_model_safety_ratio = (
        _cfg_float(wm_sampling_cfg, "SafetyCriticalRatio", 0.20)
        if _cfg_bool(wm_sampling_cfg, "EnableSafetyCriticalSampling", True)
        else 0.0
    )

    timings = defaultdict(list)
    sample_counts = defaultdict(int)
    update_counts = defaultdict(int)
    last_dual_kl = 0.0

    for cycle_idx in range(int(cycles)):
        step = int(phase_step) + cycle_idx * int(conf.JointTrainAgent.TrainAgentEverySteps)

        for _ in range(int(args.model_updates)):
            start = timer.now()
            batch = _sample_batch(
                replay_buffer,
                args.batch_size,
                args.batch_length,
                safety_critical_ratio=world_model_safety_ratio,
                high_cost_threshold=high_cost_threshold,
                boundary_low=boundary_low,
                boundary_high=boundary_high,
            )
            timings[f"{record_prefix}/sample_world_model"].append(timer.elapsed(start))
            sample_counts["world_model"] += 1

            start = timer.now()
            train_world_model_step_dfd_v4(batch, world_model, agent, None, step)
            timings[f"{record_prefix}/world_model_update"].append(timer.elapsed(start))
            update_counts["world_model"] += 1

        if _cfg_bool(gp_cfg, "Enable", True):
            for _ in range(int(args.gp_updates)):
                start = timer.now()
                batch = _sample_batch(
                    replay_buffer,
                    args.batch_size,
                    args.batch_length,
                    safety_critical_ratio=_cfg_float(gp_cfg, "SafetyCriticalRatio", 0.20),
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                )
                timings[f"{record_prefix}/sample_gp"].append(timer.elapsed(start))
                sample_counts["gp"] += 1

                start = timer.now()
                gp_critic.update(batch, world_model, agent, dual_policy, logger=None, step=step)
                timings[f"{record_prefix}/gp_update"].append(timer.elapsed(start))
                update_counts["gp"] += 1

        if _cfg_bool(gd_cfg, "Enable", True):
            for _ in range(int(args.gd_updates)):
                start = timer.now()
                batch = _sample_batch(
                    replay_buffer,
                    args.batch_size,
                    args.batch_length,
                    safety_critical_ratio=_cfg_float(gd_cfg, "SafetyCriticalRatio", 0.40),
                    high_cost_threshold=high_cost_threshold,
                    boundary_low=boundary_low,
                    boundary_high=boundary_high,
                )
                timings[f"{record_prefix}/sample_gd"].append(timer.elapsed(start))
                sample_counts["gd"] += 1

                start = timer.now()
                gd_critic.update(batch, world_model, dual_policy, logger=None, step=step)
                timings[f"{record_prefix}/gd_update"].append(timer.elapsed(start))
                update_counts["gd"] += 1

        if _cfg_bool(dual_update_cfg, "Enable", True) and int(step) >= _cfg_int(dual_update_cfg, "StartStep", 100000):
            for _ in range(int(args.dual_updates)):
                start = timer.now()
                batch = _sample_batch(replay_buffer, args.batch_size, args.batch_length)
                timings[f"{record_prefix}/sample_dual"].append(timer.elapsed(start))
                sample_counts["dual"] += 1

                start = timer.now()
                info_dual = update_dual_v4(
                    batch,
                    world_model,
                    agent,
                    gd_critic,
                    dual_policy,
                    dual_update_cfg,
                    cost_cfg=cost_cfg,
                    logger=None,
                    step=step,
                )
                timings[f"{record_prefix}/dual_update"].append(timer.elapsed(start))
                if info_dual:
                    last_dual_kl = abs(float(info_dual.get("kl_to_main", 0.0)))
                update_counts["dual"] += 1

        for _ in range(int(args.agent_updates)):
            start = timer.now()
            imagine_samples = replay_buffer.sample(args.imagine_batch_size, args.imagine_context)
            timings[f"{record_prefix}/sample_agent"].append(timer.elapsed(start))
            sample_counts["agent"] += 1

            start = timer.now()
            train_agent_step_dfd_v4(
                imagine_samples,
                world_model,
                agent,
                gp_critic,
                args.imagine_horizon,
                None,
                step,
                fdpi_cfg=fdpi_cfg,
            )
            timings[f"{record_prefix}/agent_update"].append(timer.elapsed(start))
            update_counts["agent"] += 1

    for key in list(timings):
        if timings[key]:
            torch.cuda.empty_cache() if "cuda" in str(args.device) and torch.cuda.is_available() else None

    per_key_sum = {key.split("/", 1)[1]: _sum(value) for key, value in timings.items()}
    per_key_mean = {key.split("/", 1)[1]: _mean(value) for key, value in timings.items()}
    sample_seconds = sum(value for key, value in per_key_sum.items() if key.startswith("sample_"))
    update_seconds = sum(value for key, value in per_key_sum.items() if key.endswith("_update"))
    total_seconds = sample_seconds + update_seconds
    return {
        "phase_step": int(phase_step),
        "cycles": int(cycles),
        "cycle_env_steps": int(conf.JointTrainAgent.TrainAgentEverySteps),
        "seconds": per_key_sum,
        "mean_seconds": per_key_mean,
        "sample_seconds": float(sample_seconds),
        "update_seconds": float(update_seconds),
        "total_seconds": float(total_seconds),
        "seconds_per_cycle": float(total_seconds / max(int(cycles), 1)),
        "env_steps_per_sec_updates_only": float(
            int(conf.JointTrainAgent.TrainAgentEverySteps) * int(cycles) / max(total_seconds, 1.0e-9)
        ),
        "sample_counts": {key: int(value) for key, value in sample_counts.items()},
        "update_counts": {key: int(value) for key, value in update_counts.items()},
        "last_dual_kl": float(last_dual_kl),
    }


def _estimate_training_time(conf, rollout_result, phase_results):
    num_envs = int(conf.JointTrainAgent.NumEnvs)
    train_every_steps = int(conf.JointTrainAgent.TrainAgentEverySteps)
    sample_max_steps = int(conf.JointTrainAgent.SampleMaxSteps)
    warmup_steps = int(51200)
    dual_start = int(conf.FDPIRegimeDreamer.DualUpdate.StartStep)
    fdpi_start = int(conf.FDPIRegimeDreamer.MainFDPIRegime.StartStep)
    rollout_iter_seconds = float(rollout_result["total_seconds"]) / max(int(rollout_result["iters"]), 1)
    rollout_seconds_per_cycle = rollout_iter_seconds * max(train_every_steps // num_envs, 1)

    phase_by_step = {int(item["phase_step"]): item for item in phase_results}
    available = sorted(phase_by_step)

    def nearest_phase(step):
        candidates = [value for value in available if value <= step]
        if candidates:
            return phase_by_step[candidates[-1]]
        return phase_by_step[available[0]]

    segments = [
        ("warmup_no_update", 0, min(warmup_steps, sample_max_steps), None),
        ("wm_gp_gd_agent_no_dual", warmup_steps, min(dual_start, sample_max_steps), nearest_phase(0)),
        ("dual_on_fdpi_off", dual_start, min(fdpi_start, sample_max_steps), nearest_phase(dual_start)),
        ("dual_on_fdpi_on", fdpi_start, sample_max_steps, nearest_phase(fdpi_start)),
    ]
    output_segments = []
    total_seconds = 0.0
    for name, start, end, phase in segments:
        steps = max(int(end) - int(start), 0)
        if steps <= 0:
            continue
        rollout_seconds = (steps / max(num_envs, 1)) * rollout_iter_seconds
        update_seconds = 0.0
        if phase is not None:
            cycles = steps / max(train_every_steps, 1)
            update_seconds = cycles * float(phase["seconds_per_cycle"])
        seconds = rollout_seconds + update_seconds
        total_seconds += seconds
        output_segments.append(
            {
                "name": name,
                "start": int(start),
                "end": int(end),
                "steps": int(steps),
                "phase_step_used": None if phase is None else int(phase["phase_step"]),
                "rollout_seconds": float(rollout_seconds),
                "update_seconds": float(update_seconds),
                "seconds": float(seconds),
                "env_steps_per_sec": float(steps / max(seconds, 1.0e-9)),
            }
        )
    return {
        "sample_max_steps": int(sample_max_steps),
        "rollout_seconds_per_vector_step": float(rollout_iter_seconds),
        "rollout_seconds_per_update_cycle": float(rollout_seconds_per_cycle),
        "segments": output_segments,
        "total_seconds": float(total_seconds),
        "total_hours": float(total_seconds / 3600.0),
        "overall_env_steps_per_sec": float(sample_max_steps / max(total_seconds, 1.0e-9)),
    }


def _write_report(save_dir, payload):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "dfd_v5_speed_benchmark.json")
    md_path = os.path.join(save_dir, "dfd_v5_speed_benchmark.md")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)

    lines = [
        "# DFD v5 Speed Benchmark",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- hardware: `{payload['hardware']}`",
        f"- num_envs: `{payload['config']['num_envs']}`",
        f"- batch: `{payload['config']['batch_size']} x {payload['config']['batch_length']}`",
        f"- imagine: batch `{payload['config']['imagine_batch_size']}`, context `{payload['config']['imagine_context']}`, horizon `{payload['config']['imagine_horizon']}`",
        "",
        "## Rollout",
        "",
        f"- policy rollout throughput: `{payload['policy_rollout']['env_steps_per_sec']:.2f}` env steps/s",
        f"- one vector step: `{payload['estimate']['rollout_seconds_per_vector_step']:.4f}` s",
        "",
        "## Update Phases",
        "",
        "| phase_step | sec/cycle | updates-only env steps/s | sample sec | update sec |",
        "|---:|---:|---:|---:|---:|",
    ]
    for phase in payload["phases"]:
        lines.append(
            f"| {phase['phase_step']} | {phase['seconds_per_cycle']:.4f} | "
            f"{phase['env_steps_per_sec_updates_only']:.2f} | {phase['sample_seconds']:.4f} | {phase['update_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Estimated Full Run",
            "",
            f"- sample_max_steps: `{payload['estimate']['sample_max_steps']}`",
            f"- total time: `{payload['estimate']['total_hours']:.2f}` h",
            f"- overall throughput: `{payload['estimate']['overall_env_steps_per_sec']:.2f}` env steps/s",
            "",
            "| segment | steps | phase | rollout h | update h | total h | env steps/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seg in payload["estimate"]["segments"]:
        phase = "none" if seg["phase_step_used"] is None else str(seg["phase_step_used"])
        lines.append(
            f"| {seg['name']} | {seg['steps']} | {phase} | "
            f"{seg['rollout_seconds'] / 3600.0:.3f} | {seg['update_seconds'] / 3600.0:.3f} | "
            f"{seg['seconds'] / 3600.0:.3f} | {seg['env_steps_per_sec']:.2f} |"
        )
    lines.extend(["", f"JSON: `{json_path}`", ""])
    with open(md_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Short DFD v5 training-speed benchmark.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--env_name", default="Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--imagine_batch_size", type=int, default=64)
    parser.add_argument("--imagine_context", type=int, default=16)
    parser.add_argument("--imagine_horizon", type=int, default=15)
    parser.add_argument("--prefill_rows", type=int, default=80)
    parser.add_argument("--prefill_margin_rows", type=int, default=8)
    parser.add_argument("--rollout_iters", type=int, default=16)
    parser.add_argument("--dual_rollout_ratio", type=float, default=0.18)
    parser.add_argument("--warmup_cycles", type=int, default=1)
    parser.add_argument("--bench_cycles", type=int, default=2)
    parser.add_argument("--phase_steps", type=int, nargs="*", default=[0, 200000, 2000000])
    parser.add_argument("--model_updates", type=int, default=4)
    parser.add_argument("--gp_updates", type=int, default=2)
    parser.add_argument("--gd_updates", type=int, default=2)
    parser.add_argument("--dual_updates", type=int, default=2)
    parser.add_argument("--agent_updates", type=int, default=4)
    parser.add_argument("--disable_force_head", action="store_true")
    parser.add_argument("--no_obs_normalizer", action="store_true")
    parser.add_argument("--save_root", default="eval_results/dfd_v5_speed_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    print("[benchmark] launching IsaacLab", flush=True)
    train_v5._launch_isaac(headless=True)
    print("[benchmark] loading training deps", flush=True)
    train_v5._load_training_deps()
    train_v5.torch.backends.cudnn.benchmark = False
    train_v5.seed_np_torch(seed=args.seed)

    conf = train_v5.load_dfd_v5_config(args.config_path)
    _set_config_overrides(conf, args)
    timer = Timer(train_v5.torch, args.device)
    vec_env = None
    try:
        print("[benchmark] building env/models/replay", flush=True)
        vec_env, replay_buffer, world_model, agent, gp_critic, gd_critic, dual_policy, obs_dim, action_dim = _build_everything(
            args,
            conf,
        )
        print("[benchmark] pre-filling replay", flush=True)
        prefill = _prefill_replay(args, conf, vec_env, replay_buffer, action_dim, timer)
        if not replay_buffer.can_sample(args.batch_length):
            raise RuntimeError(
                f"Replay prefill did not produce sampleable windows: rows={replay_buffer.length + 1}, "
                f"batch_length={args.batch_length}"
            )
        print("[benchmark] measuring policy rollout", flush=True)
        policy_rollout = _measure_policy_rollout(
            args,
            conf,
            vec_env,
            replay_buffer,
            world_model,
            agent,
            gp_critic,
            dual_policy,
            action_dim,
            timer,
        )

        phase_results = []
        for phase_step in args.phase_steps:
            print(f"[benchmark] measuring phase step={phase_step}", flush=True)
            if args.warmup_cycles > 0:
                _benchmark_phase(
                    args=args,
                    conf=conf,
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    agent=agent,
                    gp_critic=gp_critic,
                    gd_critic=gd_critic,
                    dual_policy=dual_policy,
                    phase_step=int(phase_step),
                    cycles=int(args.warmup_cycles),
                    timer=timer,
                    record_prefix=f"warmup_{phase_step}",
                )
            phase = _benchmark_phase(
                args=args,
                conf=conf,
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                gp_critic=gp_critic,
                gd_critic=gd_critic,
                dual_policy=dual_policy,
                phase_step=int(phase_step),
                cycles=int(args.bench_cycles),
                timer=timer,
                record_prefix=f"phase_{phase_step}",
            )
            phase_results.append(phase)

        estimate = _estimate_training_time(conf, policy_rollout, phase_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.abspath(os.path.join(args.save_root, timestamp))
        payload = {
            "timestamp": timestamp,
            "hardware": _hardware_summary(),
            "elapsed_wall_seconds_including_startup": float(time.perf_counter() - started),
            "config_path": os.path.abspath(args.config_path),
            "env_name": args.env_name,
            "config": {
                "num_envs": int(conf.JointTrainAgent.NumEnvs),
                "batch_size": int(args.batch_size),
                "batch_length": int(args.batch_length),
                "imagine_batch_size": int(args.imagine_batch_size),
                "imagine_context": int(args.imagine_context),
                "imagine_horizon": int(args.imagine_horizon),
                "model_updates": int(args.model_updates),
                "gp_updates": int(args.gp_updates),
                "gd_updates": int(args.gd_updates),
                "dual_updates": int(args.dual_updates),
                "agent_updates": int(args.agent_updates),
                "force_head": bool(conf.ForceHead.Enable),
                "obs_normalizer": bool(conf.Models.WorldModel.ObsNormalizer.Enable),
                "use_amp": bool(conf.BasicSettings.UseAmp),
                "sample_max_steps": int(conf.JointTrainAgent.SampleMaxSteps),
                "train_every_steps": int(conf.JointTrainAgent.TrainAgentEverySteps),
            },
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "prefill": prefill,
            "policy_rollout": policy_rollout,
            "phases": phase_results,
            "estimate": estimate,
        }
        json_path, md_path = _write_report(save_dir, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved JSON: {json_path}")
        print(f"Saved report: {md_path}")
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        if train_v5.simulation_app is not None:
            train_v5.simulation_app.close()


if __name__ == "__main__":
    main()
