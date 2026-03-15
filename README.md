# ChromaHack

ChromaHack is a local research repo for studying reward hacking with a fragile visual proxy. The PPO agent optimizes a CNN-based proxy reward, while evaluation tracks the hidden true reward based on real object placement.

## Project layout

The package source of truth now lives under `chromahack/`:

- `chromahack.envs.chroma_env`: Gymnasium environment
- `chromahack.data.generate_dataset`: synthetic dataset generation
- `chromahack.data.inspect_dataset`: dataset inspection
- `chromahack.models.reward_cnn`: proxy CNN models and checkpoint helpers
- `chromahack.training.train_proxy_cnn`: proxy training
- `chromahack.training.train_ppo`: PPO training
- `chromahack.evaluation.eval_hidden`: hidden-reward evaluation
- `chromahack.intervention.preference_reward_model`: preference-based intervention
- `chromahack.run_experiment`: phase orchestration

The legacy root files are preserved as thin wrappers, so existing commands like `python train_proxy_cnn.py` still work.

## Install

```bash
python -m pip install -r requirements.txt
```

## Canonical commands

Generate a dataset:

```bash
python -m chromahack.data.generate_dataset --fragility high --visualize --out_dir artifacts/data_high
```

Inspect a dataset:

```bash
python -m chromahack.data.inspect_dataset --dataset artifacts/data_high/dataset.pkl
```

Train the proxy CNN:

```bash
python -m chromahack.training.train_proxy_cnn --mode tiny --dataset_dir artifacts/data_high --out_dir artifacts/proxy_tiny
```

Train PPO from an existing proxy:

```bash
python -m chromahack.training.train_ppo --mode tiny --proxy_path artifacts/proxy_tiny/proxy_cnn.pth --total_steps 200000 --out_dir runs/exp_001
```

Evaluate hidden reward:

```bash
python -m chromahack.evaluation.eval_hidden --model_dir runs/exp_001 --n_episodes 50
```

Run the preference intervention:

```bash
python -m chromahack.intervention.preference_reward_model collect --agent_path runs/exp_001/ppo_final.zip --traj_dir artifacts/prefs/traj
python -m chromahack.intervention.preference_reward_model label --traj_dir artifacts/prefs/traj --n_pairs 2000
python -m chromahack.intervention.preference_reward_model train --traj_dir artifacts/prefs/traj --out_dir artifacts/prefs/model
python -m chromahack.intervention.preference_reward_model retrain --pref_model_path artifacts/prefs/model/pref_reward.pth --out_dir runs/exp_aligned
python -m chromahack.evaluation.eval_hidden --model_dir runs/exp_aligned --model_name ppo_aligned_final
```

Run the full experiment:

```bash
python -m chromahack.run_experiment --phase AB --quick
```

The `--quick` preset is tuned for this CPU-only local environment and defaults to `--n_envs 1`.

Smoke test:

```bash
python scripts/smoke_test.py
```

## Compatibility notes

- `dataset.pkl` is now stored as plain serializable payload data, not pickled custom objects.
- Dataset loaders still accept legacy payloads containing old `DatasetSample` or `DatasetStats` objects.
- PPO training can either bootstrap a proxy internally or load one with `--proxy_path`.
- Hidden evaluation now supports `--model_name`, defaulting to `ppo_final`.
