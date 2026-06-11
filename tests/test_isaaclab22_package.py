import os
import sys
import unittest

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ISAACLAB22_ROOT = os.path.join(PROJECT_ROOT, "fdpi_reachability_dreamer_isaaclab22")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class IsaacLab22PackageTests(unittest.TestCase):
    def test_runtime_imports_resolve_inside_isaaclab22_package(self):
        import fdpi_reachability_dreamer_isaaclab22.cost_utils as cost_utils
        import fdpi_reachability_dreamer_isaaclab22.env_wrapper as env_wrapper
        import fdpi_reachability_dreamer_isaaclab22.risk_critics as risk_critics
        import fdpi_reachability_dreamer_isaaclab22.trainer as trainer
        import fdpi_reachability_dreamer_isaaclab22.trainer_base as trainer_base
        import fdpi_reachability_dreamer_isaaclab22.world_model as world_model

        for module in (cost_utils, env_wrapper, risk_critics, trainer, trainer_base, world_model):
            self.assertTrue(os.path.abspath(module.__file__).startswith(ISAACLAB22_ROOT), module.__file__)

    def test_fdpi_replay_start_cache_is_invalidated_on_append(self):
        from fdpi_reachability_dreamer_isaaclab22.replay_buffer import FDPIReplayBuffer

        num_envs = 4
        replay = FDPIReplayBuffer(
            obs_dim=3,
            action_dim=2,
            num_envs=num_envs,
            max_length=64,
            warmup_length=0,
            device="cpu",
        )
        for step in range(6):
            replay.append(
                torch.full((num_envs, 3), float(step)),
                torch.zeros(num_envs, 2),
                torch.zeros(num_envs),
                torch.zeros(num_envs, dtype=torch.bool),
                torch.zeros(num_envs),
            )

        self.assertTrue(replay.can_sample(4))
        self.assertGreater(len(replay._start_cache), 0)

        batch = replay.sample(
            8,
            4,
            return_dict=True,
            safety_critical_ratio=0.5,
            high_cost_threshold=0.1,
        )
        self.assertEqual(tuple(batch["obs"].shape), (8, 4, 3))
        self.assertEqual(tuple(batch["action"].shape), (8, 4, 2))

        replay.append(
            torch.zeros(num_envs, 3),
            torch.zeros(num_envs, 2),
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs),
        )
        self.assertEqual(replay._start_cache, {})

    def test_fdpi_replay_safety_ratio_uses_full_batch_budget(self):
        from fdpi_reachability_dreamer_isaaclab22.cost_utils import SOURCE_DUAL
        from fdpi_reachability_dreamer_isaaclab22.replay_buffer import FDPIReplayBuffer

        num_envs = 4
        replay = FDPIReplayBuffer(
            obs_dim=3,
            action_dim=2,
            num_envs=num_envs,
            max_length=64,
            warmup_length=0,
            device="cpu",
        )
        for step in range(6):
            replay.append(
                torch.full((num_envs, 3), float(step)),
                torch.zeros(num_envs, 2),
                torch.zeros(num_envs),
                torch.zeros(num_envs, dtype=torch.bool),
                torch.zeros(num_envs),
                source=torch.full((num_envs, 1), SOURCE_DUAL, dtype=torch.int64),
            )

        counts = replay._safety_counts_by_env(
            horizon=4,
            per_env_batch=1,
            safety_critical_ratio=0.5,
            high_cost_threshold=0.1,
            boundary_low=0.05,
            boundary_high=0.4,
        )
        self.assertEqual(sum(counts), 2)

        batch = replay.sample(
            num_envs,
            4,
            return_dict=True,
            safety_critical_ratio=0.5,
            high_cost_threshold=0.1,
        )
        self.assertEqual(tuple(batch["obs"].shape), (num_envs, 4, 3))

    def test_fdpi_replay_sample_many_matches_batch_shapes_and_fields(self):
        from fdpi_reachability_dreamer_isaaclab22.cost_utils import SOURCE_DUAL
        from fdpi_reachability_dreamer_isaaclab22.replay_buffer import FDPIReplayBuffer

        num_envs = 4
        replay = FDPIReplayBuffer(
            obs_dim=3,
            action_dim=2,
            num_envs=num_envs,
            max_length=64,
            warmup_length=0,
            device="cpu",
        )
        for step in range(8):
            replay.append(
                torch.full((num_envs, 3), float(step)),
                torch.zeros(num_envs, 2),
                torch.zeros(num_envs),
                torch.zeros(num_envs, dtype=torch.bool),
                torch.zeros(num_envs),
                source=torch.full((num_envs, 1), SOURCE_DUAL, dtype=torch.int64),
            )

        batches = replay.sample_many(
            3,
            8,
            4,
            return_dict=True,
            safety_critical_ratio=0.5,
            high_cost_threshold=0.1,
        )
        self.assertEqual(len(batches), 3)
        expected_keys = {
            "obs",
            "action",
            "reward",
            "done",
            "is_first",
            "continuous_cost",
            "binary_cost",
            "extreme_cost",
            "bottom_force",
            "force_excess",
            "source",
            "cost",
        }
        for batch in batches:
            self.assertTrue(expected_keys.issubset(batch.keys()))
            self.assertEqual(tuple(batch["obs"].shape), (8, 4, 3))
            self.assertEqual(tuple(batch["action"].shape), (8, 4, 2))
            self.assertEqual(tuple(batch["source"].shape), (8, 4, 1))

        tuple_batches = replay.sample_many(2, 8, 4, return_dict=False)
        self.assertEqual(len(tuple_batches), 2)
        self.assertEqual(tuple(tuple_batches[0][0].shape), (8, 4, 3))


if __name__ == "__main__":
    unittest.main()
