from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    cfg_get,
    disable_optimizer_dynamo_wrappers,
    ensure_optimizer_step_no_grad,
    posterior_features,
    unwrap_optimizer_step,
)
from .modules import networks as net
from .sampling import source_cost_weight


TARGET_TD_BINARY = "td_binary"
TARGET_N_STEP_REACHABILITY = "n_step_reachability_td"


def _flat_sample(policy, feat: torch.Tensor) -> torch.Tensor:
    prefix_shape = feat.shape[:-1]
    flat_feat = feat.reshape(-1, feat.shape[-1])
    action = policy.sample(flat_feat)
    return action.reshape(*prefix_shape, action.shape[-1])


@torch.no_grad()
def compute_n_step_reachability_target(
    *,
    cost_window: torch.Tensor,
    done_window: torch.Tensor,
    z_boot: torch.Tensor,
    main_policy,
    target_critic,
    gamma: float,
    horizon: int,
    risk_max: float,
) -> torch.Tensor:
    """Compute strict n-step reachability TD targets.

    cost_window and done_window have shape [B, T, H + 1, 1]. The bootstrap
    state is z_{t+H+1}; future costs and bootstrap terms are masked whenever
    a done boundary is crossed.
    """
    horizon = int(horizon)
    if cost_window.ndim != 4 or done_window.ndim != 4:
        raise ValueError("cost_window and done_window must have shape [B, T, H + 1, 1].")
    if cost_window.shape[:3] != done_window.shape[:3]:
        raise ValueError(
            "cost_window and done_window shape mismatch: "
            f"{tuple(cost_window.shape)} vs {tuple(done_window.shape)}"
        )
    if cost_window.shape[2] != horizon + 1:
        raise ValueError(f"cost_window length must equal H + 1 ({horizon + 1}), got {cost_window.shape[2]}.")
    if z_boot.shape[:2] != cost_window.shape[:2]:
        raise ValueError(
            "z_boot batch/time shape must match target windows: "
            f"{tuple(z_boot.shape[:2])} vs {tuple(cost_window.shape[:2])}"
        )

    gamma = float(gamma)
    risk_max = float(risk_max)
    cost_window = cost_window.clamp(0.0, risk_max)
    done_window = done_window.clamp(0.0, 1.0)

    boot_action = _flat_sample(main_policy, z_boot)
    boot_risk = target_critic.target_reduce(
        target_critic.target_critic1(z_boot, boot_action),
        target_critic.target_critic2(z_boot, boot_action),
    ).clamp(0.0, risk_max)

    alive = torch.ones_like(cost_window[:, :, 0])
    cost_target = torch.zeros_like(alive)
    for k in range(horizon + 1):
        candidate = (gamma**k) * cost_window[:, :, k] * alive
        cost_target = torch.maximum(cost_target, candidate)
        alive = alive * (1.0 - done_window[:, :, k])

    boot_target = (gamma ** (horizon + 1)) * boot_risk * alive
    return torch.maximum(cost_target, boot_target).clamp(0.0, risk_max)


class LatentRiskCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, hidden_dim, num_layers, act):
        super().__init__()
        layers = []
        last_dim = int(feat_dim) + int(action_dim)
        for _ in range(int(num_layers)):
            layers.append(net.FeedForwardLayer(last_dim, int(hidden_dim), act()))
            last_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 1)
        self.head.apply(net.uniform_weight_init(0.0))

    def forward(self, feat, action):
        x = torch.cat((feat, action), dim=-1)
        if len(self.backbone) > 0:
            x = self.backbone(x)
        return self.head(x)


class _DoubleRiskCritic(nn.Module):
    prefix = "Risk"
    critic_prefix = "risk"
    target_reduce = staticmethod(torch.minimum)
    policy_reduce = staticmethod(torch.minimum)

    def __init__(
        self,
        feat_dim,
        action_dim,
        hidden_dim,
        num_layers,
        gamma_cost,
        target_tau,
        risk_max,
        lr,
        eps,
        use_amp,
        act,
        device,
        max_grad_norm=100.0,
        source_aware_weight=True,
        dual_source_weight=1.0,
        high_cost_weight=1.0,
        boundary_weight=1.0,
        high_cost_threshold=0.1,
        boundary_low=0.05,
        boundary_high=0.4,
        cost_key="continuous_cost",
        target_type=TARGET_TD_BINARY,
        reachability_h=5,
        reachability_gamma=0.97,
        use_reachability_weight=False,
        reachability_positive_weight=1.0,
        reachability_positive_threshold=0.5,
    ):
        super().__init__()
        self.device = device
        self.gamma_cost = float(gamma_cost)
        self.target_tau = float(target_tau)
        self.risk_max = float(risk_max)
        self.max_grad_norm = max_grad_norm
        self.source_aware_weight = bool(source_aware_weight)
        self.dual_source_weight = float(dual_source_weight)
        self.high_cost_weight = float(high_cost_weight)
        self.boundary_weight = float(boundary_weight)
        self.high_cost_threshold = float(high_cost_threshold)
        self.boundary_low = float(boundary_low)
        self.boundary_high = float(boundary_high)
        self.cost_key = str(cost_key or "continuous_cost")
        self.target_type = str(target_type or TARGET_TD_BINARY)
        self.reachability_h = max(int(reachability_h), 0)
        self.reachability_gamma = float(reachability_gamma)
        self.use_reachability_weight = bool(use_reachability_weight)
        self.reachability_positive_weight = float(reachability_positive_weight)
        self.reachability_positive_threshold = float(reachability_positive_threshold)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = bool(use_amp)

        self.critic1 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.critic2 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.to(device)
        for module in (self.target_critic1, self.target_critic2):
            for param in module.parameters():
                param.requires_grad_(False)

        disable_optimizer_dynamo_wrappers()
        self.optimizer = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
            eps=eps,
        )
        unwrap_optimizer_step(self.optimizer)
        ensure_optimizer_step_no_grad(self.optimizer)

    @classmethod
    def from_config(cls, feat_dim, action_dim, cfg, *, use_amp, act, device, default_lr=1.0e-4, default_eps=1.0e-8):
        return cls(
            feat_dim=feat_dim,
            action_dim=action_dim,
            hidden_dim=int(cfg_get(cfg, "HiddenDim", cfg_get(cfg, "hidden_dim", 256))),
            num_layers=int(cfg_get(cfg, "NumLayers", cfg_get(cfg, "num_layers", 2))),
            gamma_cost=float(cfg_get(cfg, "GammaCost", 0.97)),
            target_tau=float(cfg_get(cfg, "TargetTau", 0.005)),
            risk_max=float(cfg_get(cfg, "RiskMax", 1.0)),
            lr=float(cfg_get(cfg, "LR", default_lr)),
            eps=float(cfg_get(cfg, "Eps", default_eps)),
            use_amp=use_amp,
            act=act,
            device=device,
            max_grad_norm=float(cfg_get(cfg, "GradClipNorm", 100.0)),
            source_aware_weight=bool(cfg_get(cfg, "SourceAwareWeight", True)),
            dual_source_weight=float(cfg_get(cfg, "DualSourceWeight", 1.0)),
            high_cost_weight=float(cfg_get(cfg, "HighCostWeight", 1.0)),
            boundary_weight=float(cfg_get(cfg, "BoundaryWeight", 1.0)),
            high_cost_threshold=float(cfg_get(cfg, "HighCostThreshold", 0.1)),
            boundary_low=float(cfg_get(cfg, "BoundaryLow", 0.05)),
            boundary_high=float(cfg_get(cfg, "BoundaryHigh", 0.4)),
            cost_key=str(cfg_get(cfg, "CostKey", "continuous_cost")),
            target_type=str(cfg_get(cfg, "TargetType", TARGET_TD_BINARY)),
            reachability_h=int(cfg_get(cfg, "ReachabilityH", 5)),
            reachability_gamma=float(cfg_get(cfg, "ReachabilityGamma", cfg_get(cfg, "GammaCost", 0.97))),
            use_reachability_weight=bool(cfg_get(cfg, "UseReachabilityWeight", False)),
            reachability_positive_weight=float(cfg_get(cfg, "ReachabilityPositiveWeight", 1.0)),
            reachability_positive_threshold=float(cfg_get(cfg, "ReachabilityPositiveThreshold", 0.5)),
        ).to(device)

    @torch.no_grad()
    def soft_update_targets(self):
        for source, target in ((self.critic1, self.target_critic1), (self.critic2, self.target_critic2)):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)
            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(source_buffer)

    def risk(self, feat, action, clamp=True):
        value = self.policy_reduce(self.critic1(feat, action), self.critic2(feat, action))
        if clamp:
            value = value.clamp(0.0, self.risk_max)
        return value

    @torch.no_grad()
    def risk_no_grad(self, feat, action, clamp=True):
        return self.risk(feat, action, clamp=clamp)

    def _aligned_latent_transition_batch(self, batch, world_model):
        obs = batch["obs"].to(world_model.device)
        action = batch["action"].to(world_model.device)
        cost = batch.get(self.cost_key)
        if cost is None:
            cost = batch.get("continuous_cost", batch.get("cost"))
        cost = cost.to(world_model.device)
        done = batch["done"].to(world_model.device)
        source = batch["source"].to(world_model.device)
        is_first = batch["is_first"].to(world_model.device)

        feat = posterior_features(world_model, obs, action, is_first)
        if feat.shape[1] < 2:
            raise ValueError(f"{self.prefix} update needs at least two posterior latent steps.")

        z = feat[:, :-1]
        z_next = feat[:, 1:]
        replay_action = action[:, 1 : 1 + z.shape[1]]
        replay_cost = cost[:, 1 : 1 + z.shape[1]]
        replay_done = done[:, 1 : 1 + z.shape[1]]
        replay_source = source[:, 1 : 1 + z.shape[1]]
        return (
            z.flatten(0, 1).detach(),
            replay_action.flatten(0, 1).detach(),
            z_next.flatten(0, 1).detach(),
            replay_cost.flatten(0, 1).detach().clamp(0.0, self.risk_max),
            replay_done.flatten(0, 1).detach().clamp(0.0, 1.0),
            replay_source.flatten(0, 1).detach().to(torch.int64),
        )

    def _aligned_latent_transition_sequence(self, batch, world_model, *, horizon: int):
        obs = batch["obs"].to(world_model.device)
        action = batch["action"].to(world_model.device)
        cost = batch.get(self.cost_key)
        if cost is None:
            cost = batch.get("continuous_cost", batch.get("cost"))
        cost = cost.to(world_model.device)
        done = batch["done"].to(world_model.device)
        source = batch["source"].to(world_model.device)
        is_first = batch["is_first"].to(world_model.device)

        feat = posterior_features(world_model, obs, action, is_first)
        horizon = int(horizon)
        valid_steps = feat.shape[1] - horizon - 1
        if valid_steps <= 0:
            raise ValueError(
                f"{self.prefix} n-step update needs at least H+2 posterior latent steps; "
                f"got feat_len={feat.shape[1]}, H={horizon}."
            )

        z = feat[:, :valid_steps]
        z_boot = feat[:, horizon + 1 : horizon + 1 + valid_steps]
        replay_action = action[:, 1 : 1 + valid_steps]
        replay_source = source[:, 1 : 1 + valid_steps]
        cost_window = torch.stack(
            [cost[:, 1 + k : 1 + k + valid_steps] for k in range(horizon + 1)],
            dim=2,
        )
        done_window = torch.stack(
            [done[:, 1 + k : 1 + k + valid_steps] for k in range(horizon + 1)],
            dim=2,
        )
        return (
            z.detach(),
            replay_action.detach(),
            z_boot.detach(),
            cost_window.detach().clamp(0.0, self.risk_max),
            done_window.detach().clamp(0.0, 1.0),
            replay_source.detach().to(torch.int64),
        )

    def _weights(self, cost, source):
        if not self.source_aware_weight:
            return torch.ones_like(cost)
        return source_cost_weight(
            cost,
            source,
            high_cost_weight=self.high_cost_weight,
            dual_source_weight=self.dual_source_weight,
            boundary_weight=self.boundary_weight,
            high_cost_threshold=self.high_cost_threshold,
            boundary_low=self.boundary_low,
            boundary_high=self.boundary_high,
        )

    def _reachability_weights(self, cost, source, target):
        weight = self._weights(cost, source)
        if not self.use_reachability_weight:
            return weight
        reach_mask = (cost <= self.high_cost_threshold) & (target > self.reachability_positive_threshold)
        return torch.where(reach_mask, weight * self.reachability_positive_weight, weight)

    def _update_impl(self, batch, world_model, next_policy, *, logger=None, step=None, main_policy=None, dual_policy=None):
        self.train()
        world_model.eval()
        if hasattr(next_policy, "eval"):
            next_policy.eval()
        z, action, z_next, cost, done, source = self._aligned_latent_transition_batch(batch, world_model)
        weight = self._weights(cost, source)

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_action = next_policy.sample(z_next)
                target_risk = self.target_reduce(
                    self.target_critic1(z_next, next_action),
                    self.target_critic2(z_next, next_action),
                )
                y = cost + (1.0 - done) * self.gamma_cost * target_risk
                y = y.clamp(0.0, self.risk_max)

            pred1 = self.critic1(z, action)
            pred2 = self.critic2(z, action)
            loss1_per = weight * (pred1 - y).pow(2)
            loss2_per = weight * (pred2 - y).pow(2)
            loss = loss1_per.mean() + loss2_per.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
        else:
            grad_norms = [param.grad.detach().float().norm(2) for param in critic_params if param.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms)) if grad_norms else loss.new_tensor(0.0)
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().float().item())
        with torch.no_grad():
            self.optimizer.step()
        self.soft_update_targets()

        with torch.no_grad():
            risk = self.policy_reduce(pred1, pred2).clamp(0.0, self.risk_max)
            high_mask = cost > self.high_cost_threshold
            low_mask = ~high_mask
            dual_mask = source == SOURCE_DUAL
            main_mask = source == SOURCE_MAIN
            high_mean = risk[high_mask].mean() if high_mask.any() else risk.new_tensor(0.0)
            low_mean = risk[low_mask].mean() if low_mask.any() else risk.new_tensor(0.0)
            source_dual_loss = (loss1_per[dual_mask] + loss2_per[dual_mask]).mean() if dual_mask.any() else risk.new_tensor(0.0)
            source_main_loss = (loss1_per[main_mask] + loss2_per[main_mask]).mean() if main_mask.any() else risk.new_tensor(0.0)
            main_action_mean = risk.new_tensor(0.0)
            dual_action_mean = risk.new_tensor(0.0)
            if main_policy is not None:
                main_action = main_policy.sample(z)
                main_action_mean = self.risk(z, main_action).mean()
            if dual_policy is not None:
                dual_action = dual_policy.sample(z)
                dual_action_mean = self.risk(z, dual_action).mean()
        info = {
            "loss": float(loss.detach().float().item()),
            "mean": float(risk.detach().float().mean().item()),
            "high_cost_mean": float(high_mean.detach().float().item()),
            "low_cost_mean": float(low_mean.detach().float().item()),
            "separation": float((high_mean - low_mean).detach().float().item()),
            "target_type": 0.0,
            "reachability_h": float(self.reachability_h),
            "reachability_gamma": float(self.reachability_gamma),
            "target_mean": float(y.detach().float().mean().item()),
            "target_max": float(y.detach().float().max().item()),
            "target_positive_rate": float((y > self.reachability_positive_threshold).float().mean().item()),
            "reachability_positive_rate": 0.0,
            "reachability_positive_weighted_mass": 0.0,
            "cost_t_weighted_mass": float(
                (weight[high_mask].sum() / weight.sum().clamp_min(1.0e-6)).detach().float().item()
            ),
            "source_dual_loss": float(source_dual_loss.detach().float().item()),
            "source_main_loss": float(source_main_loss.detach().float().item()),
            "main_action_mean": float(main_action_mean.detach().float().item()),
            "dual_action_mean": float(dual_action_mean.detach().float().item()),
            "grad_norm": grad_norm_value,
            "cost_key_binary": float(self.cost_key == "binary_cost"),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"{self.prefix}/{key}", value, step)
        return info


class GpReachabilityCritic(_DoubleRiskCritic):
    """Gp critic with optional n-step reachability TD targets."""

    prefix = "Gp"
    critic_prefix = "gp"
    target_reduce = staticmethod(torch.maximum)
    policy_reduce = staticmethod(torch.maximum)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gp1 = self.critic1
        self.gp2 = self.critic2
        self.target_gp1 = self.target_critic1
        self.target_gp2 = self.target_critic2

    def update(self, batch, world_model, main_policy, dual_policy=None, *, logger=None, step=None):
        if self.target_type == TARGET_N_STEP_REACHABILITY:
            return self._update_n_step_reachability(
                batch,
                world_model,
                main_policy,
                dual_policy=dual_policy,
                logger=logger,
                step=step,
            )
        if self.target_type != TARGET_TD_BINARY:
            raise ValueError(f"Unsupported Gp TargetType: {self.target_type}")
        return self._update_impl(
            batch,
            world_model,
            main_policy,
            logger=logger,
            step=step,
            main_policy=main_policy,
            dual_policy=dual_policy,
        )

    def _update_n_step_reachability(self, batch, world_model, main_policy, dual_policy=None, *, logger=None, step=None):
        self.train()
        world_model.eval()
        if hasattr(main_policy, "eval"):
            main_policy.eval()
        z_seq, action_seq, z_boot_seq, cost_window, done_window, source_seq = self._aligned_latent_transition_sequence(
            batch,
            world_model,
            horizon=self.reachability_h,
        )
        cost_t_seq = cost_window[:, :, 0]

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                y_seq = compute_n_step_reachability_target(
                    cost_window=cost_window,
                    done_window=done_window,
                    z_boot=z_boot_seq,
                    main_policy=main_policy,
                    target_critic=self,
                    gamma=self.reachability_gamma,
                    horizon=self.reachability_h,
                    risk_max=self.risk_max,
                )
            z = z_seq.flatten(0, 1)
            action = action_seq.flatten(0, 1)
            cost = cost_t_seq.flatten(0, 1)
            source = source_seq.flatten(0, 1)
            y = y_seq.flatten(0, 1)
            weight = self._reachability_weights(cost, source, y)

            pred1 = self.critic1(z, action)
            pred2 = self.critic2(z, action)
            loss1_per = weight * (pred1 - y).pow(2)
            loss2_per = weight * (pred2 - y).pow(2)
            loss = loss1_per.mean() + loss2_per.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
        else:
            grad_norms = [param.grad.detach().float().norm(2) for param in critic_params if param.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms)) if grad_norms else loss.new_tensor(0.0)
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().float().item())
        with torch.no_grad():
            self.optimizer.step()
        self.soft_update_targets()

        with torch.no_grad():
            risk = self.policy_reduce(pred1, pred2).clamp(0.0, self.risk_max)
            high_mask = cost > self.high_cost_threshold
            low_mask = ~high_mask
            reach_mask = (cost <= self.high_cost_threshold) & (y > self.reachability_positive_threshold)
            dual_mask = source == SOURCE_DUAL
            main_mask = source == SOURCE_MAIN
            high_mean = risk[high_mask].mean() if high_mask.any() else risk.new_tensor(0.0)
            low_mean = risk[low_mask].mean() if low_mask.any() else risk.new_tensor(0.0)
            source_dual_loss = (loss1_per[dual_mask] + loss2_per[dual_mask]).mean() if dual_mask.any() else risk.new_tensor(0.0)
            source_main_loss = (loss1_per[main_mask] + loss2_per[main_mask]).mean() if main_mask.any() else risk.new_tensor(0.0)
            main_action = main_policy.sample(z)
            main_action_mean = self.risk(z, main_action).mean()
            dual_action_mean = risk.new_tensor(0.0)
            if dual_policy is not None:
                dual_action = dual_policy.sample(z)
                dual_action_mean = self.risk(z, dual_action).mean()
            weight_sum = weight.sum().clamp_min(1.0e-6)
            reachability_positive_weighted_mass = weight[reach_mask].sum() / weight_sum
            cost_t_weighted_mass = weight[high_mask].sum() / weight_sum
        info = {
            "loss": float(loss.detach().float().item()),
            "mean": float(risk.detach().float().mean().item()),
            "high_cost_mean": float(high_mean.detach().float().item()),
            "low_cost_mean": float(low_mean.detach().float().item()),
            "separation": float((high_mean - low_mean).detach().float().item()),
            "target_type": 1.0,
            "reachability_h": float(self.reachability_h),
            "reachability_gamma": float(self.reachability_gamma),
            "target_mean": float(y.detach().float().mean().item()),
            "target_max": float(y.detach().float().max().item()),
            "target_positive_rate": float((y > self.reachability_positive_threshold).float().mean().item()),
            "reachability_positive_rate": float(reach_mask.float().mean().item()),
            "reachability_positive_weighted_mass": float(
                reachability_positive_weighted_mass.detach().float().item()
            ),
            "cost_t_weighted_mass": float(cost_t_weighted_mass.detach().float().item()),
            "source_dual_loss": float(source_dual_loss.detach().float().item()),
            "source_main_loss": float(source_main_loss.detach().float().item()),
            "main_action_mean": float(main_action_mean.detach().float().item()),
            "dual_action_mean": float(dual_action_mean.detach().float().item()),
            "grad_norm": grad_norm_value,
            "cost_key_binary": float(self.cost_key == "binary_cost"),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"{self.prefix}/{key}", value, step)
        return info


GpRiskCritic = GpReachabilityCritic
GpRiskCriticV5 = GpReachabilityCritic


class GdRiskCritic(_DoubleRiskCritic):
    """Double continuous-risk critic for dual-policy continuation risk."""

    prefix = "Gd"
    critic_prefix = "gd"
    target_reduce = staticmethod(torch.minimum)
    policy_reduce = staticmethod(torch.minimum)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gd1 = self.critic1
        self.gd2 = self.critic2
        self.target_gd1 = self.target_critic1
        self.target_gd2 = self.target_critic2

    def update(self, batch, world_model, dual_policy, *, logger=None, step=None):
        return self._update_impl(
            batch,
            world_model,
            dual_policy,
            logger=logger,
            step=step,
            dual_policy=dual_policy,
        )
