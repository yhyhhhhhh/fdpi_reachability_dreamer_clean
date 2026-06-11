from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "2026-06-11_4090训练加速参数对比")
DEFAULT_ISAACLAB_SH = "/root/IsaacLab/isaaclab.sh"


def _load_yaml(path):
    import yaml

    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin) or {}
    base_path = data.pop("BaseConfig", data.pop("_BASE_", None))
    if not base_path:
        return data
    base_path = os.path.expanduser(str(base_path))
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_path)
    base = _load_yaml(base_path)
    _deep_update(base, data)
    return base


def _deep_update(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_abs(path, base=PROJECT_ROOT):
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base, path))


def _cfg_get(mapping, keys, default=None):
    node = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _relaunch_under_isaaclab(config_path, extra_args):
    if os.environ.get("FDPI_PROFILING_UNDER_ISAACLAB") == "1":
        return False
    config = _load_yaml(config_path)
    launcher = _cfg_get(config, ("运行", "isaaclab_sh"), DEFAULT_ISAACLAB_SH)
    launcher = _as_abs(launcher)
    if not os.path.isfile(launcher):
        raise FileNotFoundError(f"IsaacLab launcher not found: {launcher}")
    env = os.environ.copy()
    env["FDPI_PROFILING_UNDER_ISAACLAB"] = "1"
    env["WANDB_MODE"] = str(_cfg_get(config, ("日志", "wandb_mode"), "disabled"))
    env["TERM"] = env.get("TERM") or "xterm"
    final_root = PROJECT_ROOT
    isaaclab_root = os.path.abspath(os.path.dirname(launcher))
    surgical_robot5_ext = str(
        _cfg_get(
            config,
            ("运行", "surgical_robot5_ext"),
            "/root/gpufree-data/surgical_robot5/exts/surgical_robot5",
        )
    )
    path_parts = [
        os.path.join(isaaclab_root, "source/isaaclab"),
        os.path.join(isaaclab_root, "source/isaaclab_tasks"),
        os.path.join(isaaclab_root, "source/isaaclab_assets"),
        os.path.join(isaaclab_root, "source/isaaclab_rl"),
        surgical_robot5_ext,
        final_root,
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(part for part in path_parts if part)
    cmd = [launcher, "-p", os.path.abspath(__file__), "--config", os.path.abspath(config_path), "--_inner"]
    cmd.extend(extra_args)
    print("[profiling] relaunching under IsaacLab:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
    raise SystemExit(result.returncode)


class CudaTimer:
    def __init__(self, torch_module, device):
        self.torch = torch_module
        self.device = str(device)

    def sync(self):
        if "cuda" in self.device and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def now(self):
        self.sync()
        return time.perf_counter()

    def elapsed(self, start):
        self.sync()
        return time.perf_counter() - start


class TimedLocalLogger:
    def __init__(self, timer):
        self.timer = timer
        self.tot_step = -1
        self.log_dict = {}
        self.seconds = 0.0
        self.count = 0
        self.records = []

    def log(self, tag, value, step):
        start = self.timer.now()
        try:
            if step > self.tot_step:
                if self.log_dict:
                    self.records.append({"step": int(self.tot_step), "values": dict(self.log_dict)})
                self.log_dict = {}
                self.tot_step = int(step)
            self.log_dict[str(tag)] = _json_safe(value)
        finally:
            self.seconds += self.timer.elapsed(start)
            self.count += 1

    def flush(self):
        if self.log_dict:
            self.records.append({"step": int(self.tot_step), "values": dict(self.log_dict)})
            self.log_dict = {}


def _json_safe(value):
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().float().item())
            return float(value.detach().float().mean().item())
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _sum(values):
    return float(sum(values))


def _mean(values):
    return float(sum(values) / max(len(values), 1))


def _cfg_float(train_mod, node, name, default):
    return float(train_mod.cfg_get(node, name, default))


def _cfg_bool(train_mod, node, name, default=False):
    return bool(train_mod.cfg_get(node, name, default))


def _cfg_int(train_mod, node, name, default):
    return int(train_mod.cfg_get(node, name, default))


def _cfg_int_tuple(train_mod, node, name, default=()):
    value = train_mod.cfg_get(node, name, default)
    if value is None:
        return tuple()
    return tuple(int(v) for v in value)


def _set_config_overrides(train_mod, conf, profile_cfg, output_dirs):
    conf.defrost()
    conf.JointTrainAgent.NumEnvs = int(profile_cfg["num_envs"])
    conf.JointTrainAgent.BatchSize = int(profile_cfg["batch_size"])
    conf.JointTrainAgent.BatchLength = int(profile_cfg["batch_length"])
    conf.JointTrainAgent.ImagineBatchSize = int(profile_cfg["imagine_batch_size"])
    conf.JointTrainAgent.ImagineContext = int(profile_cfg["imagine_context"])
    conf.JointTrainAgent.ImagineHorizon = int(profile_cfg["imagine_horizon"])
    conf.JointTrainAgent.BufferWarmUp = int(profile_cfg["buffer_warmup_rows"]) * int(profile_cfg["num_envs"])
    conf.JointTrainAgent.SaveEverySteps = 10**12
    if hasattr(conf, "Env") and hasattr(conf.Env, "MakeKwargs"):
        conf.Env.MakeKwargs.num_envs = int(profile_cfg["num_envs"])
    if bool(profile_cfg.get("disable_obs_normalizer", False)):
        conf.Models.WorldModel.ObsNormalizer.Enable = False
    if bool(profile_cfg.get("save_replay_buffer", False)) is False:
        conf.FDPIRegimeDreamer.Checkpoint.SaveReplayBuffer = False
    train_mod._validate_batch_config(conf)
    conf.freeze()
    os.makedirs(output_dirs["checkpoint_dir"], exist_ok=True)


def _build_everything(train_mod, args_ns, conf, profile_cfg):
    train_args = SimpleNamespace(
        env_name=args_ns.env_name,
        device=args_ns.device,
        seed=args_ns.seed,
        num_envs=None,
    )
    vec_env = train_mod.build_env(train_args, conf)
    obs_dim = int(vec_env.single_observation_space["policy"].shape[0])
    action_dim = int(vec_env.single_action_space.shape[0])
    act = getattr(train_mod.nn, conf.Models.Act)
    world_model = train_mod.build_world_model(conf, obs_dim, action_dim, act, args_ns.device)
    agent = train_mod.build_agent(conf, action_dim, act, args_ns.device)
    gp_critic = train_mod.build_gp_critic(conf, action_dim, act, args_ns.device)
    gd_critic = train_mod.build_gd_critic(conf, action_dim, act, args_ns.device)
    dual_policy = train_mod.build_dual_policy(conf, action_dim, act, args_ns.device)
    if bool(train_mod.cfg_get(conf.FDPIRegimeDreamer.DualPolicy, "InitFromMainActor", True)):
        dual_policy.initialize_from_main_actor(agent)
    replay_buffer = train_mod.DFDV4ReplayBuffer(
        obs_dim,
        action_dim,
        vec_env.num_envs,
        int(profile_cfg["buffer_max_length"]),
        int(profile_cfg["buffer_warmup_rows"]) * vec_env.num_envs,
        args_ns.device,
        include_force=bool(conf.ForceHead.Enable),
        force_dim=1,
        force_key=conf.ForceHead.Key,
    )
    return vec_env, replay_buffer, world_model, agent, gp_critic, gd_critic, dual_policy, obs_dim, action_dim


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


def _append_transition(train_mod, replay_buffer, current_obs_dict, current_obs, action, reward, done, is_first, next_obs_dict, info, source, cost_cfg, device):
    from fdpi_reachability_dreamer_isaaclab22.trainer_base import _extract_force_obs
    from fdpi_reachability_dreamer_isaaclab22.cost_utils import extract_continuous_cost

    num_envs = replay_buffer.num_envs
    cost_parts = extract_continuous_cost(
        info,
        next_obs_dict,
        num_envs=num_envs,
        device=device,
        force_threshold=_cfg_float(train_mod, cost_cfg, "ForceThreshold", 0.1),
        low_force_scale=_cfg_float(train_mod, cost_cfg, "LowForceScale", 0.05),
        cost_force_max=_cfg_float(train_mod, cost_cfg, "CostForceMax", 15.0),
        force_scale=_cfg_float(train_mod, cost_cfg, "ForceScale", 5.0),
        extreme_force_threshold=_cfg_float(train_mod, cost_cfg, "ExtremeForceThreshold", 5.0),
        clip_cost=_cfg_bool(train_mod, cost_cfg, "ClipCost", True),
        cost_min=_cfg_float(train_mod, cost_cfg, "CostMin", 0.0),
        cost_max=_cfg_float(train_mod, cost_cfg, "CostMax", 1.0),
        force_key=getattr(replay_buffer, "force_key", ""),
        cost_source=str(train_mod.cfg_get(cost_cfg, "CostSource", "bottom")),
        bottom_force_channels=_cfg_int_tuple(train_mod, cost_cfg, "BottomForceChannels", [2, 5]),
        wall_force_channels=_cfg_int_tuple(train_mod, cost_cfg, "WallForceChannels", [1, 4]),
        cost_force_channels=_cfg_int_tuple(train_mod, cost_cfg, "CostForceChannels", ()),
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


def _prefill_replay(train_mod, vec_env, replay_buffer, action_dim, args_ns, conf, profile_cfg, timer, timelines):
    torch = train_mod.torch
    from fdpi_reachability_dreamer_isaaclab22.cost_utils import SOURCE_RANDOM
    from fdpi_reachability_dreamer_isaaclab22.trainer_base import _is_first, _policy_obs, _reset_after_step

    cost_cfg = conf.FDPIRegimeDreamer.ContinuousCost
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(args_ns.device)
    is_first = _is_first(current_obs_dict, vec_env.num_envs, args_ns.device)
    source = torch.full((vec_env.num_envs, 1), SOURCE_RANDOM, dtype=torch.int64, device=args_ns.device)
    warmup_rows = int(profile_cfg.get("warmup_env_steps", 0)) // max(int(vec_env.num_envs), 1)
    target_rows = max(
        int(profile_cfg["prefill_rows"]),
        warmup_rows,
        int(profile_cfg["batch_length"]) + int(profile_cfg["prefill_margin_rows"]),
    )
    timings = defaultdict(list)
    while replay_buffer.length + 1 < target_rows:
        step_start = timer.now()
        start = timer.now()
        action = _random_action(vec_env, torch, vec_env.num_envs, action_dim, args_ns.device)
        timings["policy_inference_time"].append(timer.elapsed(start))

        start = timer.now()
        next_obs_dict, reward, done, info = vec_env.step(action.detach().cpu().numpy())
        reward = torch.as_tensor(reward, dtype=torch.float32, device=args_ns.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=args_ns.device)
        timings["env_step_time"].append(timer.elapsed(start))

        start = timer.now()
        _append_transition(
            train_mod,
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
            args_ns.device,
        )
        timings["replay_insert_time"].append(timer.elapsed(start))

        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args_ns.device)
        timings["total_train_step_time"].append(timer.elapsed(step_start))
        env_steps = (replay_buffer.length + 1) * vec_env.num_envs
        timelines.append(
            {
                "phase": "warmup_unmeasured",
                "env_steps": int(env_steps),
                "env_steps_per_second": float(vec_env.num_envs / max(timings["total_train_step_time"][-1], 1.0e-9)),
                "train_updates_per_second": 0.0,
                "samples_per_second": 0.0,
                "total_train_step_time": float(timings["total_train_step_time"][-1]),
            }
        )
    return timings


def _sample_batch(replay_buffer, batch_size, horizon, **kwargs):
    return replay_buffer.sample(batch_size, horizon, return_dict=True, **kwargs)


def _benchmark_update_cycle(
    train_mod,
    conf,
    replay_buffer,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    logger,
    timer,
    args_ns,
    profile_cfg,
    phase_step,
):
    from fdpi_reachability_dreamer_isaaclab22.dual_update import update_dual
    from fdpi_reachability_dreamer_isaaclab22.trainer import train_agent_step, train_world_model_step

    fdpi_cfg = conf.FDPIRegimeDreamer
    wm_sampling_cfg = fdpi_cfg.WorldModelSampling
    gp_cfg = fdpi_cfg.Gp
    gd_cfg = fdpi_cfg.Gd
    dual_update_cfg = fdpi_cfg.DualUpdate
    cost_cfg = fdpi_cfg.ContinuousCost
    batch_size = int(profile_cfg["batch_size"])
    batch_length = int(profile_cfg["batch_length"])
    imagine_batch_size = int(profile_cfg["imagine_batch_size"])
    imagine_context = int(profile_cfg["imagine_context"])
    imagine_horizon = int(profile_cfg["imagine_horizon"])

    high_cost_threshold = _cfg_float(train_mod, wm_sampling_cfg, "HighCostThreshold", _cfg_float(train_mod, gp_cfg, "HighCostThreshold", 0.1))
    boundary_low = _cfg_float(train_mod, wm_sampling_cfg, "BoundaryLow", _cfg_float(train_mod, gp_cfg, "BoundaryLow", 0.05))
    boundary_high = _cfg_float(train_mod, wm_sampling_cfg, "BoundaryHigh", _cfg_float(train_mod, gp_cfg, "BoundaryHigh", 0.4))
    world_model_safety_ratio = (
        _cfg_float(train_mod, wm_sampling_cfg, "SafetyCriticalRatio", 0.20)
        if _cfg_bool(train_mod, wm_sampling_cfg, "EnableSafetyCriticalSampling", True)
        else 0.0
    )
    timings = defaultdict(list)
    sample_count = 0
    update_count = 0

    for _ in range(int(profile_cfg["model_updates"])):
        start = timer.now()
        batch = _sample_batch(
            replay_buffer,
            batch_size,
            batch_length,
            safety_critical_ratio=world_model_safety_ratio,
            high_cost_threshold=high_cost_threshold,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
        )
        timings["sample_batch_time"].append(timer.elapsed(start))
        sample_count += batch_size * batch_length

        start = timer.now()
        train_world_model_step(batch, world_model, agent, logger, phase_step)
        timings["world_model_update_time"].append(timer.elapsed(start))
        update_count += 1

    if _cfg_bool(train_mod, gp_cfg, "Enable", True):
        for _ in range(int(profile_cfg["gp_updates"])):
            start = timer.now()
            batch = _sample_batch(
                replay_buffer,
                batch_size,
                batch_length,
                safety_critical_ratio=_cfg_float(train_mod, gp_cfg, "SafetyCriticalRatio", 0.20),
                high_cost_threshold=high_cost_threshold,
                boundary_low=boundary_low,
                boundary_high=boundary_high,
            )
            timings["sample_batch_time"].append(timer.elapsed(start))
            sample_count += batch_size * batch_length

            start = timer.now()
            gp_critic.update(batch, world_model, agent, dual_policy, logger=logger, step=phase_step)
            timings["actor_critic_update_time"].append(timer.elapsed(start))
            update_count += 1

    if _cfg_bool(train_mod, gd_cfg, "Enable", True):
        for _ in range(int(profile_cfg["gd_updates"])):
            start = timer.now()
            batch = _sample_batch(
                replay_buffer,
                batch_size,
                batch_length,
                safety_critical_ratio=_cfg_float(train_mod, gd_cfg, "SafetyCriticalRatio", 0.40),
                high_cost_threshold=high_cost_threshold,
                boundary_low=boundary_low,
                boundary_high=boundary_high,
            )
            timings["sample_batch_time"].append(timer.elapsed(start))
            sample_count += batch_size * batch_length

            start = timer.now()
            gd_critic.update(batch, world_model, dual_policy, logger=logger, step=phase_step)
            timings["actor_critic_update_time"].append(timer.elapsed(start))
            update_count += 1

    if _cfg_bool(train_mod, dual_update_cfg, "Enable", True) and int(phase_step) >= _cfg_int(train_mod, dual_update_cfg, "StartStep", 100000):
        for _ in range(int(profile_cfg["dual_updates"])):
            start = timer.now()
            batch = _sample_batch(replay_buffer, batch_size, batch_length)
            timings["sample_batch_time"].append(timer.elapsed(start))
            sample_count += batch_size * batch_length

            start = timer.now()
            update_dual(batch, world_model, agent, gd_critic, dual_policy, dual_update_cfg, cost_cfg=cost_cfg, logger=logger, step=phase_step)
            timings["actor_critic_update_time"].append(timer.elapsed(start))
            update_count += 1

    for _ in range(int(profile_cfg["agent_updates"])):
        start = timer.now()
        imagine_samples = replay_buffer.sample(imagine_batch_size, imagine_context)
        timings["sample_batch_time"].append(timer.elapsed(start))
        sample_count += imagine_batch_size * imagine_context

        start = timer.now()
        train_agent_step(imagine_samples, world_model, agent, gp_critic, imagine_horizon, logger, phase_step, fdpi_cfg=fdpi_cfg)
        timings["actor_critic_update_time"].append(timer.elapsed(start))
        update_count += 1

    return timings, sample_count, update_count


def _measure_policy_rollout(
    train_mod,
    conf,
    vec_env,
    replay_buffer,
    world_model,
    agent,
    gp_critic,
    gd_critic,
    dual_policy,
    logger,
    timer,
    args_ns,
    profile_cfg,
    timelines,
):
    torch = train_mod.torch
    from fdpi_reachability_dreamer_isaaclab22.trainer import _sample_policy_action
    from fdpi_reachability_dreamer_isaaclab22.trainer_base import _is_first, _policy_obs, _reset_after_step

    cost_cfg = conf.FDPIRegimeDreamer.ContinuousCost
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(args_ns.device)
    is_first = _is_first(current_obs_dict, vec_env.num_envs, args_ns.device)
    state = world_model.initial(vec_env.num_envs)
    timings = defaultdict(list)
    sample_total = 0
    update_total = 0
    env_steps_total = 0

    for iter_idx in range(int(profile_cfg["rollout_iters"])):
        step_start = timer.now()
        env_steps = int(iter_idx * vec_env.num_envs)

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
                use_dual_sampling=float(profile_cfg["dual_rollout_ratio"]) > 0.0,
                dual_ratio=float(profile_cfg["dual_rollout_ratio"]),
                num_envs=vec_env.num_envs,
                device=args_ns.device,
            )
        timings["policy_inference_time"].append(timer.elapsed(start))

        start = timer.now()
        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=args_ns.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=args_ns.device)
        timings["env_step_time"].append(timer.elapsed(start))

        start = timer.now()
        _append_transition(
            train_mod,
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
            args_ns.device,
        )
        timings["replay_insert_time"].append(timer.elapsed(start))

        start = timer.now()
        logger.log("Profiling/reward_mean", float(reward.detach().float().mean().item()), env_steps)
        logger.log("Profiling/buffer_length", len(replay_buffer), env_steps)
        timings["logging_time"].append(timer.elapsed(start))

        start = timer.now()
        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, args_ns.device)
        timings["reset_after_step_time"].append(timer.elapsed(start))

        if iter_idx >= int(profile_cfg["update_start_iter"]) and (iter_idx - int(profile_cfg["update_start_iter"])) % int(profile_cfg["update_every_iters"]) == 0:
            cycle_timings, sample_count, update_count = _benchmark_update_cycle(
                train_mod,
                conf,
                replay_buffer,
                world_model,
                agent,
                gp_critic,
                gd_critic,
                dual_policy,
                logger,
                timer,
                args_ns,
                profile_cfg,
                int(profile_cfg["phase_step"]),
            )
            for key, values in cycle_timings.items():
                timings[key].extend(values)
            sample_total += sample_count
            update_total += update_count

        step_seconds = timer.elapsed(step_start)
        timings["total_train_step_time"].append(step_seconds)
        env_steps_total += vec_env.num_envs
        elapsed_components = {
            key: timings[key][-1]
            for key in (
                "policy_inference_time",
                "env_step_time",
                "replay_insert_time",
                "logging_time",
                "total_train_step_time",
            )
            if timings.get(key)
        }
        update_seconds_now = sum(values[-1] for key, values in timings.items() if key.endswith("_update_time") and values)
        sample_seconds_now = timings["sample_batch_time"][-1] if timings.get("sample_batch_time") else 0.0
        timelines.append(
            {
                "phase": "rollout_update",
                "env_steps": int(env_steps_total),
                "env_steps_per_second": float(vec_env.num_envs / max(step_seconds, 1.0e-9)),
                "train_updates_per_second": float(update_total / max(sum(timings.get("total_train_step_time", [])), 1.0e-9)),
                "samples_per_second": float(sample_total / max(sum(timings.get("sample_batch_time", [])), 1.0e-9)) if sample_total else 0.0,
                "total_train_step_time": float(step_seconds),
                "component_seconds": json.dumps(elapsed_components, ensure_ascii=False),
                "latest_update_seconds": float(update_seconds_now),
                "latest_sample_seconds": float(sample_seconds_now),
            }
        )

    return timings, env_steps_total, sample_total, update_total


def _save_checkpoint(train_mod, checkpoint_dir, world_model, agent, gp_critic, gd_critic, dual_policy, replay_buffer, timer, step):
    torch = train_mod.torch
    start = timer.now()
    torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{step}.pth"))
    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_{step}.pth"))
    torch.save(gp_critic.state_dict(), os.path.join(checkpoint_dir, f"gp_{step}.pth"))
    torch.save(gd_critic.state_dict(), os.path.join(checkpoint_dir, f"gd_{step}.pth"))
    torch.save(dual_policy.state_dict(), os.path.join(checkpoint_dir, f"dual_policy_{step}.pth"))
    return timer.elapsed(start)


def _hardware_summary():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        return output
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def _start_gpu_collector(config, experiment_dir):
    interval = float(_cfg_get(config, ("GPU采样", "interval_seconds"), 1.0))
    output = _as_abs(_cfg_get(config, ("GPU采样", "output_csv"), "日志/gpu_stats.csv"), experiment_dir)
    script = os.path.join(PROJECT_ROOT, "scripts", "profiling", "collect_gpu_stats.py")
    cmd = [sys.executable, script, "--output", output, "--interval", str(interval)]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    return proc, output


def _stop_gpu_collector(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_plotter(experiment_dir):
    from scripts.profiling.plot_training_speed import plot_experiment

    return plot_experiment(experiment_dir)


def _write_experiment_record(experiment_dir, metrics, generated_images):
    summary = metrics.get("summary", {})
    bottleneck = metrics.get("bottleneck", {})
    profile_cfg = metrics.get("profile_config", {})
    training_config_path = metrics.get("training_config_path", "")
    images_md = []
    for path in generated_images:
        rel = os.path.relpath(path, experiment_dir)
        name = os.path.basename(path)
        images_md.append(f"![{name}]({rel})\n\n{name}：用于观察本次 profiling 的对应速度或 GPU 指标。")
    content = f"""# 实验记录：4090训练加速参数对比

## 1. 实验目的

- 测量当前配置在 RTX 4090 上的实际训练吞吐。
- 分开定位环境 step、策略推理、replay append / sample、world model、Gp/Gd、dual、agent 更新、日志和 checkpoint 的相对耗时。
- 对照 baseline、4090 balanced、4090 fast 三组参数，判断降低更新/采样比例、增大 batch 和启用 TF32 后是否提升 wall-clock 吞吐。

## 2. 实验配置

- 实验日期：2026-06-11
- 训练配置：`{training_config_path}`
- 运行入口：`运行命令.sh`
- profiling 配置：`配置.yaml`
- 输出目录：`{experiment_dir}`
- wandb：disabled，本次只写本地文件。
- NumEnvs：`{profile_cfg.get("num_envs", 0)}`
- BatchSize：`{profile_cfg.get("batch_size", 0)}`
- ImagineBatchSize：`{profile_cfg.get("imagine_batch_size", 0)}`
- 更新组合：WM=`{profile_cfg.get("model_updates", 0)}`，Gp=`{profile_cfg.get("gp_updates", 0)}`，Gd=`{profile_cfg.get("gd_updates", 0)}`，dual=`{profile_cfg.get("dual_updates", 0)}`，agent=`{profile_cfg.get("agent_updates", 0)}`

## 3. 代码修改

- 新增 4090 fast / balanced 配置和实验归档入口。
- 训练循环加入 timing 日志，replay sampling 增加 starts 缓存，日志频率可配置。
- 默认配置保持可回退，4090 配置通过 `BaseConfig` 覆盖参数。
- 详细改动见 `代码变更.patch`。

## 4. 核心结果

### 4.1 关键图片

{chr(10).join(images_md)}

### 4.2 关键指标

- 平均环境吞吐：`{summary.get("env_steps_per_second", 0.0):.2f}` env steps/s
- 训练更新吞吐：`{summary.get("train_updates_per_second", 0.0):.2f}` updates/s
- batch 样本吞吐：`{summary.get("samples_per_second", 0.0):.2f}` samples/s
- 平均 GPU 利用率：`{summary.get("gpu_utilization_mean_percent", 0.0):.2f}%`
- 峰值显存占用：`{summary.get("gpu_memory_used_max_mib", 0.0):.0f}` MiB
- 最大耗时阶段：`{bottleneck.get("name", "待补充")}`，占比 `{bottleneck.get("share", 0.0) * 100.0:.1f}%`

## 5. 结果分析

- 当前 profiling 显示主要耗时阶段为 `{bottleneck.get("name", "待补充")}`。
- 若 GPU 利用率长期偏低且环境 step 占比高，优先怀疑仿真/环境交互瓶颈。
- 若 GPU 利用率较高且更新阶段占比高，优先怀疑 world model 或 actor/critic 训练计算瓶颈。

## 6. 主要问题

- profiling hook 会带来少量同步开销，因此绝对速度略低于完全无 profiling 的训练。
- fast 配置优先 wall-clock 吞吐，可能牺牲一定样本效率，需要后续 1M env steps 长跑确认 reward、cost、Gp/Gd loss 不崩坏。
- 如果 fast 出现 OOM 或安全学习明显变差，优先回退到 balanced，或把 Gp/Gd/Dual UpdateSteps 从 1 恢复到 2。

## 7. 初步结论

- 本次测试给出该配置下的可复现速度、GPU 使用情况和主要耗时阶段。
- 是否采用 fast 作为正式长训参数，应以后续三组对比和 1M env steps 稳定性验证为准。

## 8. 下一步计划

- 先运行 baseline / balanced / fast 三组 profiling，记录 env steps/s、GPU 利用率和 update 占比。
- 选择吞吐和稳定性更好的配置，运行 1M env steps 长跑验证。
"""
    path = os.path.join(experiment_dir, "实验记录.md")
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(content)
    return path


def _summarize_gpu(gpu_csv):
    rows = []
    if os.path.isfile(gpu_csv):
        with open(gpu_csv, "r", encoding="utf-8") as fin:
            rows = list(csv.DictReader(fin))

    def values(key):
        out = []
        for row in rows:
            try:
                out.append(float(row.get(key, 0.0)))
            except Exception:
                pass
        return out

    util = values("gpu_utilization_percent")
    mem = values("gpu_memory_used_mib")
    mem_total = values("gpu_memory_total_mib")
    power = values("gpu_power_draw_w")
    temp = values("gpu_temperature_c")
    return {
        "gpu_samples": len(rows),
        "gpu_utilization_mean_percent": _mean(util) if util else 0.0,
        "gpu_utilization_max_percent": max(util) if util else 0.0,
        "gpu_memory_used_mean_mib": _mean(mem) if mem else 0.0,
        "gpu_memory_used_max_mib": max(mem) if mem else 0.0,
        "gpu_memory_total_mib": max(mem_total) if mem_total else 0.0,
        "gpu_power_draw_mean_w": _mean(power) if power else 0.0,
        "gpu_temperature_max_c": max(temp) if temp else 0.0,
    }


def _phase_summary(timings):
    keys = [
        "policy_inference_time",
        "env_step_time",
        "replay_insert_time",
        "sample_batch_time",
        "world_model_update_time",
        "actor_critic_update_time",
        "logging_time",
        "checkpoint_time",
        "total_train_step_time",
    ]
    phase_seconds = {key: _sum(timings.get(key, [])) for key in keys}
    phase_means = {key: _mean(timings.get(key, [])) for key in keys}
    comparable = {key: value for key, value in phase_seconds.items() if key != "total_train_step_time" and value > 0}
    total = sum(comparable.values())
    if comparable:
        name, seconds = max(comparable.items(), key=lambda item: item[1])
    else:
        name, seconds = "unknown", 0.0
    return phase_seconds, phase_means, {"name": name, "seconds": seconds, "share": float(seconds / max(total, 1.0e-9))}


def _write_code_patch(experiment_dir):
    patch_path = os.path.join(experiment_dir, "代码变更.patch")
    try:
        patch = subprocess.check_output(
            [
                "git",
                "diff",
                "--",
                "fdpi_reachability_dreamer_isaaclab22",
                "configs",
                "scripts",
                experiment_dir,
                "docs/research",
                "README.md",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        patch = (
            f"无法生成 git diff: {exc}\n\n"
            "当前工作区未提供可用 git diff 时，本文件作为变更摘要记录。\n"
            "本次 4090 加速改动涉及：\n"
            "- configs/reachability_gp_isaaclab22_4090_fast.yaml\n"
            "- configs/reachability_gp_isaaclab22_4090_balanced.yaml\n"
            "- fdpi_reachability_dreamer_isaaclab22/train.py\n"
            "- fdpi_reachability_dreamer_isaaclab22/trainer.py\n"
            "- fdpi_reachability_dreamer_isaaclab22/replay_buffer.py\n"
            "- scripts/train.sh\n"
            "- scripts/profiling/benchmark_training_speed.py\n"
            "- experiments/2026-06-11_4090训练加速参数对比/\n"
            "- README.md\n"
            "- docs/research/CHANGELOG.md\n"
            "- docs/research/EXPERIMENT_INDEX.md\n"
            "- docs/research/PROJECT_STATE.md\n"
        )
    with open(patch_path, "w", encoding="utf-8") as fout:
        fout.write(patch)
    return patch_path


def _copy_config_snapshot(config_path, experiment_dir):
    snapshot_dir = os.path.join(experiment_dir, "配置快照")
    os.makedirs(snapshot_dir, exist_ok=True)
    if os.path.isfile(config_path):
        import shutil

        target_path = os.path.join(snapshot_dir, os.path.basename(config_path))
        if os.path.abspath(config_path) != os.path.abspath(target_path):
            shutil.copy2(config_path, target_path)


def run_inner(config_path):
    import fdpi_reachability_dreamer_isaaclab22.train as train_mod

    config = _load_yaml(config_path)
    experiment_dir = _as_abs(_cfg_get(config, ("实验", "输出目录"), DEFAULT_EXPERIMENT_DIR))
    output_dirs = {
        "log_dir": _as_abs(_cfg_get(config, ("日志", "log_dir"), "日志"), experiment_dir),
        "checkpoint_dir": _as_abs(_cfg_get(config, ("日志", "checkpoint_dir"), "检查点"), experiment_dir),
        "image_dir": _as_abs(_cfg_get(config, ("日志", "image_dir"), "图片"), experiment_dir),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    profile_cfg = {
        "num_envs": int(_cfg_get(config, ("运行", "num_envs"), 64)),
        "batch_size": int(_cfg_get(config, ("训练", "batch_size"), 64)),
        "batch_length": int(_cfg_get(config, ("训练", "batch_length"), 64)),
        "imagine_batch_size": int(_cfg_get(config, ("训练", "imagine_batch_size"), 64)),
        "imagine_context": int(_cfg_get(config, ("训练", "imagine_context"), 16)),
        "imagine_horizon": int(_cfg_get(config, ("训练", "imagine_horizon"), 15)),
        "buffer_max_length": int(_cfg_get(config, ("训练", "buffer_max_length"), 1000000)),
        "buffer_warmup_rows": int(_cfg_get(config, ("Profiling", "buffer_warmup_rows"), 0)),
        "prefill_rows": int(_cfg_get(config, ("Profiling", "prefill_rows"), 96)),
        "warmup_env_steps": int(_cfg_get(config, ("Profiling", "warmup_env_steps"), 0)),
        "prefill_margin_rows": int(_cfg_get(config, ("Profiling", "prefill_margin_rows"), 8)),
        "rollout_iters": int(_cfg_get(config, ("Profiling", "rollout_iters"), 24)),
        "update_start_iter": int(_cfg_get(config, ("Profiling", "update_start_iter"), 0)),
        "update_every_iters": max(int(_cfg_get(config, ("Profiling", "update_every_iters"), 8)), 1),
        "phase_step": int(_cfg_get(config, ("Profiling", "phase_step"), 2000000)),
        "dual_rollout_ratio": float(_cfg_get(config, ("Profiling", "dual_rollout_ratio"), 0.18)),
        "model_updates": int(_cfg_get(config, ("Profiling", "model_updates"), 4)),
        "gp_updates": int(_cfg_get(config, ("Profiling", "gp_updates"), 2)),
        "gd_updates": int(_cfg_get(config, ("Profiling", "gd_updates"), 2)),
        "dual_updates": int(_cfg_get(config, ("Profiling", "dual_updates"), 2)),
        "agent_updates": int(_cfg_get(config, ("Profiling", "agent_updates"), 4)),
        "disable_obs_normalizer": bool(_cfg_get(config, ("Profiling", "disable_obs_normalizer"), False)),
        "save_replay_buffer": bool(_cfg_get(config, ("Profiling", "save_replay_buffer"), False)),
    }
    args_ns = SimpleNamespace(
        env_name=str(
            _cfg_get(
                config,
                ("运行", "env_name"),
                "SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1",
            )
        ),
        device=str(_cfg_get(config, ("运行", "device"), "cuda:0")),
        seed=int(_cfg_get(config, ("运行", "seed"), 0)),
        num_envs=None,
    )
    training_config_path = _as_abs(
        _cfg_get(config, ("运行", "training_config"), "configs/reachability_gp_isaaclab22.yaml")
    )
    _copy_config_snapshot(training_config_path, experiment_dir)
    _copy_config_snapshot(config_path, experiment_dir)

    gpu_proc = None
    gpu_csv = _as_abs(_cfg_get(config, ("GPU采样", "output_csv"), "日志/gpu_stats.csv"), experiment_dir)

    vec_env = None
    started = time.perf_counter()
    failed = False
    try:
        print("[profiling] launching IsaacLab", flush=True)
        train_mod._launch_isaac(headless=True)
        print("[profiling] loading training deps", flush=True)
        train_mod._load_training_deps()
        train_mod.seed_np_torch(seed=args_ns.seed)
        conf = train_mod.load_dfd_v5_config(training_config_path)
        train_mod._configure_torch_performance(conf)
        normalizer = train_mod.cfg_get(conf.Models.WorldModel, "ObsNormalizer", None)
        normalizer_path = str(train_mod.cfg_get(normalizer, "Path", ""))
        if bool(train_mod.cfg_get(normalizer, "Enable", False)) and normalizer_path and not os.path.isfile(normalizer_path):
            print(f"[profiling] obs normalizer missing, disabling for profiling: {normalizer_path}", flush=True)
            profile_cfg["disable_obs_normalizer"] = True
        _set_config_overrides(train_mod, conf, profile_cfg, output_dirs)
        timer = CudaTimer(train_mod.torch, args_ns.device)
        logger = TimedLocalLogger(timer)

        print("[profiling] building env/models/replay", flush=True)
        vec_env, replay_buffer, world_model, agent, gp_critic, gd_critic, dual_policy, obs_dim, action_dim = _build_everything(
            train_mod,
            args_ns,
            conf,
            profile_cfg,
        )

        timelines = []
        all_timings = defaultdict(list)
        print("[profiling] pre-filling replay", flush=True)
        prefill_timings = _prefill_replay(train_mod, vec_env, replay_buffer, action_dim, args_ns, conf, profile_cfg, timer, timelines)
        prefill_env_steps = int((replay_buffer.length + 1) * vec_env.num_envs)
        if not replay_buffer.can_sample(profile_cfg["batch_length"]):
            raise RuntimeError(
                f"Replay prefill did not produce sampleable windows: rows={replay_buffer.length + 1}, "
                f"batch_length={profile_cfg['batch_length']}"
            )
        if bool(_cfg_get(config, ("GPU采样", "enable"), True)):
            gpu_proc, gpu_csv = _start_gpu_collector(config, experiment_dir)
            time.sleep(0.5)

        print("[profiling] measuring rollout/update cycle", flush=True)
        rollout_timings, env_steps, sample_total, update_total = _measure_policy_rollout(
            train_mod,
            conf,
            vec_env,
            replay_buffer,
            world_model,
            agent,
            gp_critic,
            gd_critic,
            dual_policy,
            logger,
            timer,
            args_ns,
            profile_cfg,
            timelines,
        )
        for key, values in rollout_timings.items():
            all_timings[key].extend(values)

        if bool(_cfg_get(config, ("Profiling", "measure_checkpoint"), True)):
            print("[profiling] measuring checkpoint save", flush=True)
            checkpoint_seconds = _save_checkpoint(
                train_mod,
                output_dirs["checkpoint_dir"],
                world_model,
                agent,
                gp_critic,
                gd_critic,
                dual_policy,
                replay_buffer,
                timer,
                int(env_steps),
            )
            all_timings["checkpoint_time"].append(checkpoint_seconds)

        _stop_gpu_collector(gpu_proc)
        gpu_proc = None
        logger.flush()
        all_timings["logging_time"].append(float(logger.seconds))
        phase_seconds, phase_mean_seconds, bottleneck = _phase_summary(all_timings)
        wall_seconds = float(time.perf_counter() - started)
        gpu_summary = _summarize_gpu(gpu_csv)
        total_train_step_seconds = phase_seconds.get("total_train_step_time", 0.0)
        sample_seconds = phase_seconds.get("sample_batch_time", 0.0)
        update_seconds = phase_seconds.get("world_model_update_time", 0.0) + phase_seconds.get("actor_critic_update_time", 0.0)
        summary = {
            "env_steps_per_second": float(env_steps / max(total_train_step_seconds, 1.0e-9)),
            "train_updates_per_second": float(update_total / max(update_seconds, 1.0e-9)),
            "samples_per_second": float(sample_total / max(sample_seconds, 1.0e-9)),
            "wall_seconds": wall_seconds,
            **gpu_summary,
        }
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "experiment_dir": experiment_dir,
            "hardware": _hardware_summary(),
            "training_config_path": training_config_path,
            "profiling_config_path": os.path.abspath(config_path),
            "env_name": args_ns.env_name,
            "device": args_ns.device,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "profile_config": profile_cfg,
            "phase_seconds": phase_seconds,
            "phase_mean_seconds": phase_mean_seconds,
            "phase_counts": {key: len(values) for key, values in all_timings.items()},
            "warmup_phase_seconds": {key: _sum(values) for key, values in prefill_timings.items()},
            "warmup_phase_counts": {key: len(values) for key, values in prefill_timings.items()},
            "summary": summary,
            "bottleneck": bottleneck,
            "logger_records": len(logger.records),
            "env_steps": int(env_steps),
            "prefill_env_steps": int(prefill_env_steps),
            "measured_env_steps": int(env_steps),
            "total_profiled_env_steps": int(prefill_env_steps + env_steps),
            "train_update_count": int(update_total),
            "sample_count": int(sample_total),
        }
        metrics_path = os.path.join(experiment_dir, "指标结果.json")
        _write_json(metrics_path, payload)
        _write_json(os.path.join(output_dirs["log_dir"], "logger_records.json"), logger.records)
        timeline_path = os.path.join(output_dirs["log_dir"], "training_speed_timeline.csv")
        timeline_fields = [
            "phase",
            "env_steps",
            "env_steps_per_second",
            "train_updates_per_second",
            "samples_per_second",
            "total_train_step_time",
            "component_seconds",
            "latest_update_seconds",
            "latest_sample_seconds",
        ]
        _write_csv(timeline_path, timelines, timeline_fields)
        _write_json(os.path.join(output_dirs["log_dir"], "phase_timings.json"), {key: values for key, values in all_timings.items()})
        generated_images = _run_plotter(experiment_dir)
        _write_experiment_record(experiment_dir, payload, generated_images)
        _write_code_patch(experiment_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[profiling] metrics: {metrics_path}", flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        _stop_gpu_collector(gpu_proc)
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        if not failed and getattr(train_mod, "simulation_app", None) is not None:
            train_mod.simulation_app.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Profile FDPI Reachability Dreamer training speed.")
    parser.add_argument("--config", required=True, help="Profiling experiment YAML config.")
    parser.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def main():
    args, extra = parse_args()
    config_path = _as_abs(args.config)
    if not args._inner:
        _relaunch_under_isaaclab(config_path, extra)
    run_inner(config_path)


if __name__ == "__main__":
    main()
