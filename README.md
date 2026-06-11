# FDPI Reachability Dreamer

Standalone clean project for the FDPI reachability Dreamer algorithm.

## Project Layout

- `fdpi_reachability_dreamer_isaaclab22/`: active IsaacLab 2.2 / IsaacSim 5.0 algorithm package.
- `fdpi_reachability_dreamer_isaaclab22/train.py`: training entrypoint.
- `fdpi_reachability_dreamer_isaaclab22/risk_critics.py`: Gp reachability critic and Gd risk critic.
- `fdpi_reachability_dreamer_isaaclab22/trainer.py`: joint world-model, main-policy, dual-policy, and risk-critic training loop.
- `fdpi_reachability_dreamer_isaaclab22/world_model.py`: continuous-cost world model.
- `configs/reachability_gp_isaaclab22.yaml`: active reachability-Gp training config.
- `scripts/train.sh`: active IsaacLab 2.2 launcher for this clean project.
- `scripts/train_isaaclab22.sh`: compatibility wrapper that forwards to `scripts/train.sh`.
- `tests/test_reachability_gp.py`: CPU tests for n-step reachability targets and Gp update behavior.

Shared pieces from earlier experiments have been folded into one package with final names.

## Key Algorithm

The Gp critic uses an n-step reachability TD target:

- `Gp.TargetType: "n_step_reachability_td"`
- `Gp.CostKey: "binary_cost"`
- `Gp.ReachabilityH: 3`
- `Gp.ReachabilityGamma: 0.97`
- `Gp.UseReachabilityWeight: true`

The target is the max over discounted binary costs inside the horizon and a bootstrapped target Gp value after the horizon, with `done` boundaries respected.

## External Dependencies

- IsaacLab 2.2 / IsaacSim 5.0
- the `surgical_robot5` IsaacLab task extension
- the `isaaclab` Python environment with PyTorch, Gymnasium, yacs, tqdm, wandb, and colorama

Default local dependency paths used by `scripts/train.sh`:

- `CONDA_SH=/opt/conda/etc/profile.d/conda.sh`
- `CONDA_ENV_NAME=isaaclab`
- `ISAACLAB_ROOT=/root/IsaacLab`
- `SURGICAL_ROBOT5_EXT=/root/gpufree-data/surgical_robot5/exts/surgical_robot5`

Override any of these environment variables if your local layout changes.

## Run

```bash
cd /root/gpufree-data/fdpi_reachability_dreamer_clean
WANDB_MODE=disabled bash scripts/train.sh
```

Short smoke test:

```bash
cd /root/gpufree-data/fdpi_reachability_dreamer_clean
WANDB_MODE=disabled \
NUM_ENVS=2 \
SAMPLE_MAX_STEPS=1024 \
BUFFER_WARMUP_STEPS=64 \
SAVE_EVERY_STEPS=512 \
bash scripts/train.sh
```

Common overrides:

```bash
FULL_CHECKPOINT_PATH=/path/to/full_state_1000000.pth \
SAMPLE_MAX_STEPS=10000000 \
bash scripts/train.sh
```

4090 speed configs:

```bash
CONFIG_PATH=configs/reachability_gp_isaaclab22_4090_fast.yaml \
WANDB_MODE=disabled \
bash scripts/train.sh
```

Use `configs/reachability_gp_isaaclab22_4090_balanced.yaml` for a more conservative comparison. For replay sampling, `BatchSize` and `ImagineBatchSize` must be greater than or equal to `NumEnvs` and divisible by `NumEnvs`.

`configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml` keeps `128 env + batch 256 + TF32` but restores more agent/Gp/Gd/dual updates than fast. Use it when fast throughput is good but policy learning becomes too weak.

4090 profiling commands:

```bash
bash experiments/2026-06-11_4090训练加速参数对比/运行命令.sh
bash experiments/2026-06-11_4090训练加速参数对比/运行全部对比.sh
```

Training outputs default to `ckpt_isaaclab22/`, which is ignored by `.gitignore`.

## Test

```bash
cd /root/gpufree-data/fdpi_reachability_dreamer_clean
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
python -m unittest discover -s tests
```
