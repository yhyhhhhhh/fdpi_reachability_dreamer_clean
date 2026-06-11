import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn


FINAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FINAL_ROOT not in sys.path:
    sys.path.insert(0, FINAL_ROOT)

from fdpi_reachability_dreamer.cost_utils import extract_continuous_cost
from fdpi_reachability_dreamer.modules.world_models import ParallelWorldModel
from fdpi_reachability_dreamer.risk_critics import GpReachabilityCritic, compute_n_step_reachability_target


class _FakeDynamic:
    def __init__(self, feat_dim, action_dim):
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)

    def parallel_observe(self, embed, action, is_first):
        del is_first
        horizon = action.shape[1]
        feat = embed[..., : self.feat_dim]
        if feat.shape[1] != horizon:
            feat = feat[:, :horizon]
        post = {"feat": feat.contiguous()}
        return post, post, None, None

    def get_feat(self, state):
        return state["feat"]

    def img_step(self, state, action):
        del action
        return {"feat": state["feat"]}


class _FakeWorldModel(nn.Module):
    def __init__(self, feat_dim=4, action_dim=2):
        super().__init__()
        self.device = "cpu"
        self.device_type = "cpu"
        self.tensor_dtype = torch.float32
        self.use_amp = False
        self.dynamic = _FakeDynamic(feat_dim, action_dim)

    def encoder(self, obs):
        return obs


class _FakePolicy(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_dim = int(action_dim)

    def sample(self, feat, greedy=False):
        del greedy
        return torch.zeros(*feat.shape[:-1], self.action_dim, dtype=feat.dtype, device=feat.device)


class _FakeMainAgent(nn.Module):
    def __init__(self, feat_dim=3, action_dim=2):
        super().__init__()
        self.actor = nn.Linear(feat_dim, 2 * action_dim)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.std_offset = 0.05
        self.std_scale = 0.95


class _ConstantRisk(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, feat, action):
        return torch.full(action[..., :1].shape, self.value, dtype=feat.dtype, device=feat.device)


class _TargetCritic(nn.Module):
    target_reduce = staticmethod(torch.maximum)

    def __init__(self, value=0.0):
        super().__init__()
        self.target_critic1 = _ConstantRisk(value)
        self.target_critic2 = _ConstantRisk(value)


class ReachabilityGpTests(unittest.TestCase):
    def test_runtime_imports_resolve_inside_clean_folder(self):
        import fdpi_reachability_dreamer.trainer as trainer
        import fdpi_reachability_dreamer.trainer_base as trainer_base
        import fdpi_reachability_dreamer.world_model as world_model

        for module in (trainer, trainer_base, world_model):
            self.assertTrue(os.path.abspath(module.__file__).startswith(FINAL_ROOT), module.__file__)

    def test_nstep_target_propagates_future_cost(self):
        gamma = 0.97
        horizon = 5
        cost_window = torch.tensor([[[[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]]]])
        done_window = torch.zeros_like(cost_window)
        z_boot = torch.zeros(1, 1, 3)
        target = compute_n_step_reachability_target(
            cost_window=cost_window,
            done_window=done_window,
            z_boot=z_boot,
            main_policy=_FakePolicy(action_dim=2),
            target_critic=_TargetCritic(value=0.0),
            gamma=gamma,
            horizon=horizon,
            risk_max=1.0,
        )
        self.assertEqual(tuple(target.shape), (1, 1, 1))
        self.assertAlmostEqual(float(target[0, 0, 0]), gamma**5, places=6)

    def test_nstep_target_current_cost_is_one(self):
        gamma = 0.97
        horizon = 5
        cost_window = torch.tensor([[[[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]])
        done_window = torch.zeros_like(cost_window)
        z_boot = torch.zeros(1, 1, 3)
        target = compute_n_step_reachability_target(
            cost_window=cost_window,
            done_window=done_window,
            z_boot=z_boot,
            main_policy=_FakePolicy(action_dim=2),
            target_critic=_TargetCritic(value=0.2),
            gamma=gamma,
            horizon=horizon,
            risk_max=1.0,
        )
        self.assertAlmostEqual(float(target[0, 0, 0]), 1.0, places=6)

    def test_nstep_target_does_not_cross_done(self):
        gamma = 0.97
        horizon = 5
        cost_window = torch.tensor([[[[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]]]])
        done_window = torch.zeros_like(cost_window)
        done_window[:, :, 1] = 1.0
        z_boot = torch.zeros(1, 1, 3)
        target = compute_n_step_reachability_target(
            cost_window=cost_window,
            done_window=done_window,
            z_boot=z_boot,
            main_policy=_FakePolicy(action_dim=2),
            target_critic=_TargetCritic(value=0.9),
            gamma=gamma,
            horizon=horizon,
            risk_max=1.0,
        )
        self.assertAlmostEqual(float(target[0, 0, 0]), 0.0, places=6)

    def test_obs_normalizer_is_runtime_only_and_strict_checkpoint_compatible(self):
        def make_model(**kwargs):
            return ParallelWorldModel(
                video_log=100,
                is_proprio=True,
                obs_shape=3,
                action_dim=2,
                stoch=2,
                discrete=2,
                hidden=8,
                stem_ch=4,
                min_res=4,
                num_bin=7,
                max_bin=5,
                dyn_scale=0.75,
                rep_scale=0.15,
                val_scale=1.0,
                kl_free=1.0,
                gamma=0.99,
                lambd=0.95,
                tau=0.01,
                lr=1e-4,
                eps=1e-8,
                use_amp=False,
                act=nn.SiLU,
                device="cpu",
                **kwargs,
            )

        base = make_model()
        state = base.state_dict()
        self.assertNotIn("obs_norm_mean", state)
        self.assertNotIn("obs_norm_std", state)

        normalized = make_model(
            obs_normalizer_enabled=True,
            obs_normalizer_mean=np.array([1.0, 2.0, -1.0], dtype=np.float32),
            obs_normalizer_std=np.array([2.0, 4.0, 0.5], dtype=np.float32),
        )
        normalized.load_state_dict(state, strict=True)
        obs = torch.tensor([[3.0, 6.0, 0.0]])
        encoded = normalized.normalize_obs(obs)
        decoded = normalized.denormalize_obs(encoded)
        self.assertTrue(torch.allclose(decoded, obs))
        self.assertTrue(torch.allclose(encoded, torch.tensor([[1.0, 1.0, 2.0]])))

    def test_continuous_cost_can_select_bottom_wall_or_custom_channels(self):
        obs = {
            "force": torch.tensor(
                [
                    [0.0, 0.4, 2.0, 0.0, 0.7, 3.0],
                    [0.0, 5.0, 0.2, 0.0, 1.5, 0.1],
                ]
            )
        }
        common = dict(
            info={},
            obs_dict=obs,
            num_envs=2,
            device="cpu",
            force_threshold=0.1,
            low_force_scale=0.05,
            cost_force_max=15.0,
            bottom_force_channels=(2, 5),
            wall_force_channels=(1, 4),
        )
        bottom = extract_continuous_cost(cost_source="bottom", **common)
        wall = extract_continuous_cost(cost_source="wall", **common)
        both = extract_continuous_cost(cost_source="bottom_wall", **common)
        custom = extract_continuous_cost(
            cost_source="custom",
            cost_force_channels=(0, 3),
            **common,
        )
        self.assertTrue(torch.equal(bottom["cost_force"].view(-1), torch.tensor([3.0, 0.2])))
        self.assertTrue(torch.equal(wall["cost_force"].view(-1), torch.tensor([0.7, 5.0])))
        self.assertTrue(torch.equal(both["cost_force"].view(-1), torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.equal(custom["cost_force"].view(-1), torch.tensor([0.0, 0.0])))

    def test_gp_reachability_update_shape_and_logging(self):
        torch.manual_seed(5)
        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        policy = _FakePolicy(action_dim=2)
        batch = {
            "obs": torch.ones(2, 9, 3),
            "action": torch.zeros(2, 9, 2),
            "binary_cost": torch.zeros(2, 9, 1),
            "done": torch.zeros(2, 9, 1),
            "is_first": torch.zeros(2, 9, 1),
            "source": torch.zeros(2, 9, 1, dtype=torch.long),
        }
        batch["binary_cost"][:, 6] = 1.0
        gp = GpReachabilityCritic(
            3,
            2,
            8,
            0,
            0.97,
            0.0,
            1.0,
            1e-4,
            1e-8,
            False,
            nn.SiLU,
            "cpu",
            cost_key="binary_cost",
            target_type="n_step_reachability_td",
            reachability_h=5,
            reachability_gamma=0.97,
            use_reachability_weight=True,
            reachability_positive_weight=3.0,
            reachability_positive_threshold=0.5,
            high_cost_threshold=0.5,
        )
        gp.target_critic1 = _ConstantRisk(0.0)
        gp.target_critic2 = _ConstantRisk(0.0)
        info = gp.update(batch, world_model, policy)
        self.assertEqual(info["target_type"], 1.0)
        self.assertEqual(info["reachability_h"], 5.0)
        self.assertGreater(info["target_positive_rate"], 0.0)
        self.assertGreater(info["reachability_positive_weighted_mass"], 0.0)

    def test_isaaclab22_posterior_states_and_features_are_consistent(self):
        from fdpi_reachability_dreamer_isaaclab22.cost_utils import (
            posterior_features,
            posterior_states,
            posterior_states_and_features,
        )

        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        obs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        action = torch.zeros(2, 4, 2)
        is_first = torch.zeros(2, 4, 1)

        state, feat = posterior_states_and_features(world_model, obs, action, is_first)
        self.assertTrue(torch.equal(feat, posterior_features(world_model, obs, action, is_first)))
        self.assertTrue(torch.equal(state["feat"], posterior_states(world_model, obs, action, is_first)["feat"]))

    def test_isaaclab22_gp_update_accepts_precomputed_posterior_feat(self):
        from fdpi_reachability_dreamer_isaaclab22.cost_utils import posterior_features
        from fdpi_reachability_dreamer_isaaclab22.risk_critics import GpReachabilityCritic as IsaacGpCritic

        torch.manual_seed(7)
        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        policy = _FakePolicy(action_dim=2)
        batch = {
            "obs": torch.ones(2, 9, 3),
            "action": torch.zeros(2, 9, 2),
            "binary_cost": torch.zeros(2, 9, 1),
            "done": torch.zeros(2, 9, 1),
            "is_first": torch.zeros(2, 9, 1),
            "source": torch.zeros(2, 9, 1, dtype=torch.long),
        }
        batch["binary_cost"][:, 6] = 1.0

        def make_gp():
            return IsaacGpCritic(
                3,
                2,
                8,
                0,
                0.97,
                0.0,
                1.0,
                1e-4,
                1e-8,
                False,
                nn.SiLU,
                "cpu",
                cost_key="binary_cost",
                target_type="n_step_reachability_td",
                reachability_h=5,
                reachability_gamma=0.97,
                use_reachability_weight=True,
                reachability_positive_weight=3.0,
                reachability_positive_threshold=0.5,
                high_cost_threshold=0.5,
            )

        gp_ref = make_gp()
        gp_latent = make_gp()
        gp_latent.load_state_dict(gp_ref.state_dict())
        feat = posterior_features(world_model, batch["obs"], batch["action"], batch["is_first"])

        info_ref = gp_ref.update(batch, world_model, policy)
        info_latent = gp_latent.update(batch, world_model, policy, posterior_feat=feat)
        self.assertAlmostEqual(info_ref["loss"], info_latent["loss"], places=6)
        self.assertAlmostEqual(info_ref["target_mean"], info_latent["target_mean"], places=6)

    def test_isaaclab22_dual_update_accepts_precomputed_posterior_state(self):
        from fdpi_reachability_dreamer_isaaclab22.cost_utils import posterior_states
        from fdpi_reachability_dreamer_isaaclab22.dual_policy import DualPolicy
        from fdpi_reachability_dreamer_isaaclab22.dual_update import update_dual
        from fdpi_reachability_dreamer_isaaclab22.risk_critics import GdRiskCritic

        torch.manual_seed(11)
        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        main_agent = _FakeMainAgent(feat_dim=3, action_dim=2)
        dual_policy = DualPolicy(2, 3, 8, 0.05, 1.0, 1e-4, 1e-8, False, nn.SiLU, "cpu")
        gd_critic = GdRiskCritic(3, 2, 8, 0, 0.97, 0.0, 1.0, 1e-4, 1e-8, False, nn.SiLU, "cpu")
        batch = {
            "obs": torch.ones(2, 4, 3),
            "action": torch.zeros(2, 4, 2),
            "is_first": torch.zeros(2, 4, 1),
        }
        state = posterior_states(world_model, batch["obs"], batch["action"], batch["is_first"])
        info = update_dual(
            batch,
            world_model,
            main_agent,
            gd_critic,
            dual_policy,
            {"Type": "imagined_risk_return", "Horizon": 2, "GammaCost": 0.97},
            cost_cfg={},
            posterior_state=state,
        )
        self.assertIn("loss", info)
        self.assertIn("kl_to_main", info)


if __name__ == "__main__":
    unittest.main()
