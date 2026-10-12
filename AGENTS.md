# AGENTS guide for this repository

This file is for code agents (Cursor, Copilot, OpenAI, etc.) working in
`projeto/`, a reinforcement learning project for a 2D robotic arm using
Gym 0.21 and PyBullet.

Keep changes incremental, preserve existing behavior, and do not auto
reformat whole files.

## 1. Environment and setup

- Python version: project documentation says 3.8+, but code already uses
  builtin generics like `list[str]`, so assume at least Python 3.9.
- Dependencies: install from `requirements.txt` in this directory:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # on Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

- Core libraries: `gym==0.21`, `pybullet`, `numpy`, `stable_baselines3`,
  `matplotlib` and standard Python libs.
- Gym API is the legacy one: `reset() -> obs`, `step() -> (obs, reward,
  done, info)`. Do not switch to the newer Gymnasium API.

## 2. Build / lint / test commands

There is no packaging or build pipeline; scripts are run directly.

### Build

- No dedicated build step. For a quick syntax check on all modules:

  ```bash
  python -m compileall .
  ```

### Lint and formatting

- No project linter or formatter config files are present (no `flake8`,
  `ruff`, `black` configs detected).
- When editing, keep formatting consistent with surrounding code and
  avoid running whole-file autoformatters unless explicitly requested.
- If you must format, limit it to the lines you touch.

### Tests and quick checks

There is no `pytest` configuration; tests are small scripts under
`scripts_test/`.

- Sanity check of Gym + NumPy:

  ```bash
  python scripts_test/sanity_check.py
  ```

- Inverse kinematics + environment distance check (closest thing to a
  single focused test):

  ```bash
  python scripts_test/test_env.py
  ```

- Visual debug of tip vs target in the PyBullet GUI:

  ```bash
  python scripts_test/test_tip_visual.py
  ```

- Demo of the original PyBullet simulation (no RL):

  ```bash
  python scripts_test/demo_inicial.py
  ```

If you want to add formal unit tests, prefer `pytest`, put them under a
`tests/` directory, and keep the existing scripts working as-is.

## 3. Running experiments

- Single PPO training run (current defaults):

  ```bash
  python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4
  python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4
  ```

- Recommended training with aligned targets (per seed):

  ```bash
  python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4 --target-seed 0
  python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4 --target-seed 0
  ```

- Evaluation of trained models (matching train seeds/targets):

  ```bash
  python eval_rl.py --mode pure --seed 0 --episodes 200 --max-steps 200 --target-seed 0
  python eval_rl.py --mode pirl --seed 0 --episodes 200 --max-steps 200 --target-seed 0
  python eval_rl.py --mode both --seed 0 --episodes 200 --max-steps 200 --out results/eval_official_seed0 --target-seed 0
  ```

- Sweep over seeds for training and evaluation (legacy helper):

  ```bash
  python run_experiments_seeds.py --first-seed 0 --last-seed 4
  ```

- New helpers for sweeps with target_seeds:

  ```bash
  python run_train_target_seeds.py --first-seed 0 --last-seed 4 --steps 2000000
  python run_eval_target_seeds.py  --mode both --model-first-seed 0 --model-last-seed 4
  ```

- Analysis of scales (distance, action norm, torque, energy):

  ```bash
  python analyze_scales.py --mode both --policy trained
  ```

- Detailed per-step inspection and CSV dumps:

  ```bash
  python inspect_episodes.py --mode both --first-seed 0 --last-seed 4 \
      --episodes-per-config 20 --max-steps 200
  ```

## 4. Code layout

- `two_link_arm_env.py`: Gym environment `TwoLinkArmEnv` built on
  PyBullet, with reward modes `pure` and `pirl` and detailed info
  fields in `info`.
- `train_rl.py`: CLI that trains PPO policies and saves monitor CSVs
  and model zip files under `runs_pure/` and `runs_pirl/`.
- `eval_rl.py`: CLI that loads trained models, runs evaluation episodes,
  prints a summary table, and writes CSV plus JSON summary files.
- `analyze_scales.py`: runs policies (random or trained) to measure
  typical scales for distance, actions, torque and energy, and produces
  bar plots.
- `inspect_episodes.py`: more detailed rollouts with rich per-step
  metrics and trajectory CSVs.
- `run_experiments_seeds.py`: orchestrates training + evaluation over a
  range of seeds by spawning `train_rl.py` and `eval_rl.py` as
  subprocesses.
- `scripts_test/`: small scripts used as sanity tests and visual debug.

## 5. Python style guidelines

### Imports

- Group imports as:
  1) standard library (`os`, `sys`, `argparse`, `time`, `json`, etc.),
  2) third-party (`numpy`, `gym`, `pybullet`, `stable_baselines3`,
     `matplotlib`),
  3) local (`from two_link_arm_env import TwoLinkArmEnv`).
- Within a group, keep imports alphabetic where reasonable.
- Prefer `from module import Name` for local modules used widely inside
  a file.

### Formatting

- Indentation is 4 spaces; no tabs.
- Aim for Black-like formatting but do not reformat entire files.
- Break long argument lists over multiple lines with trailing commas.
- Use double quotes for docstrings; string literal style inside code can
  follow the existing file (mixed `'` and `"` is acceptable, do not
  churn them).

### Types

- Use standard type hints for public functions and CLIs where it adds
  clarity (examples already exist in `run_experiments_seeds.py` and
  `inspect_episodes.py`).
- Prefer builtin generics (`list[str]`, `dict[str, float]`) to
  `List[str]` unless you need compatibility with type-checkers that do
  not support them.
- For numpy arrays, use `np.ndarray` or `np.ndarray[Any, Any]` if you
  need explicit types; otherwise keep hints simple.

### Naming

- Modules and functions: `snake_case`.
- Variables: `snake_case`, descriptive but concise (`ep_ret`,
  `tau_sum`, `tip_x`).
- Classes: `CapWords` (`TwoLinkArmEnv`).
- Constants: `UPPER_SNAKE_CASE` for file-local constants
  (`EPISODES`, `STEPS`, etc.).

### Functions and modules

- Top-level scripts follow a `main()` function plus an
  `if __name__ == "__main__": main()` guard.
- Use docstrings at the module or function level when behavior is
  nontrivial, especially for scripts and analysis tools.
- Keep side effects (file system writes, heavy training loops) behind a
  function or CLI entry point.

### Error handling and logging

- For missing models or files, raise `FileNotFoundError` with a clear
  message (see `eval_rl.py`, `inspect_episodes.py`).
- For invalid CLI arguments that pass argparse but fail logical checks
  (like `first-seed > last-seed`), exit with `SystemExit` and a short
  explanation.
- Use `try/except` sparingly; most current uses are around optional
  debug overlays or best-effort metrics. Follow that pattern: swallow
  only noncritical errors.
- Logging is done with `print` in scripts; follow the existing style and
  keep messages in Portuguese where the surrounding file does.

## 6. RL and environment specifics

- The info dict returned by `TwoLinkArmEnv.step` already contains
  several fields (`distance`, `final_distance`, `tau_sum_total`,
  `episode_energy`, etc.). When extending the environment, prefer adding
  new metrics here rather than changing existing keys.
- Reward structure differs by `reward_mode` (`pure` vs `pirl`). If you
  modify rewards, keep backward compatibility where possible and update
  comments in `two_link_arm_env.py` and relevant scripts.
- Torque limits and energy accounting are important for later analysis;
  do not silently change units or conventions without updating plotting
  and inspection scripts.

## 7. Cursor and Copilot rules

- As of this version there are no `.cursor/rules/`, `.cursorrules`, or
  `.github/copilot-instructions.md` files in the repository.
- If you introduce such rules, update this section with a short summary
  so other agents know which constraints are enforced.

## 8. When adding new code

- Mirror the existing structure: small, focused scripts with clear
  CLIs; keep long numerical experiments out of library-style modules.
- Keep backward compatibility with existing command line interfaces
  where reasonable; add new flags rather than renaming old ones.
- Prefer adding new analysis scripts next to the existing ones in this
  directory or under `scripts_test/` for quick sanity checks.

## 9. Updated training / eval / plotting (2026-10)

Recent changes extended the core scripts without breaking the original
interfaces. New agents should prefer the newer helpers when possible.

- `two_link_arm_env.py`:
  - Reward modes remain `pure` and `pirl`.
  - New parameters on `TwoLinkArmEnv.__init__` (all with defaults):
    - `lambda_a`: weight on action penalty (norm of `Δtheta`).
    - `alpha_tau`: weight on torque penalty (only used in `pirl`).
    - `step_sleep`: optional `time.sleep` after `p.stepSimulation()` when
      `render=True` (used to slow down visual training).
    - `max_delta_theta`: max absolute increment per joint per step;
      defines the `Box` action limits.
    - `target_seed`: separate RNG for target sampling, independent of
      `np.random` global; use to align target sequences across
      runs/modes.
    - `success_bonus`: scalar added to reward only on the step where
      `dist < success_threshold`.
  - `info` dict still contains distance, torque, energy and episode
    accumulators; new keys are added instead of renaming/removing.

- `train_rl.py` (single-run PPO training):
  - Existing flags kept: `--mode`, `--steps`, `--seed`, `--run-tag`,
    `--lambda-a`, `--alpha-tau`, `--learning-rate`.
  - New flags wired into `TwoLinkArmEnv`:
    - `--render`, `--step-sleep`: enable PyBullet GUI and slow it down.
    - `--target-seed`: passes through to env for deterministic target
      sequences; use the same value for pure/pirl to compare fairly.
    - `--max-delta-theta`: controls the action range in radians.
    - `--success-bonus`: enables the success bonus term in reward.
  - For each run, a `train_cmd.txt` is written under the run directory
    with the full command line and JSON dump of args.

- `eval_rl.py` (evaluation):
  - Still supports `--mode {pure,pirl,both}`, `--seed`, `--episodes`,
    `--max-steps`, `--render`, `--out`, `--run-tag`.
  - New flag `--target-seed` mirrors the env parameter: use the same
    `target_seed` as in training when you want aligned targets, or a
    different range (via `run_eval_target_seeds.py`) for cross-eval.
  - Outputs per-episode CSV plus a `_summary.json` with aggregate
    metrics used by plotting scripts.

- Helper scripts for sweeps and plotting (new):
  - `run_train_target_seeds.py`: trains `pure` and `pirl` for an interval
    of seeds, using the same integer as both PPO seed and
    `target_seed` (one run per mode×seed).
  - `run_eval_target_seeds.py`: evaluates models over a range of
    `model_seeds`, optionally paired with a different range of
    `target_seeds` (for cross-generalization).
  - `plot_train_compare.py`: reads `monitor.csv` files and produces
    per-mode training curves (reward + success) and success-per-seed
    figures; `run_plots_train.py` wraps this for a seed range.
  - `plot_eval_compare.py`: reads the `_summary.json` files from
    `eval_rl.py` and generates comparison plots pure vs pirl per seed
    (success rate, final distance, torque effort, energy, plus a 2×2
    grid figure). `run_plots_eval_all.py` orchestrates this and can
    optionally re-run `analyze_scales.py` per seed.
  - `run_plots_results_paper.py`: high-level pipeline intended for
    dissertation figures. It calls, in order:
    1) `run_plots_train.py` (training curves),
    2) `run_plots_eval_all.py` (pure vs pirl eval),
    3) `scripts_graficos/script_plot_train.py` (aggregate train PDFs),
    4) `scripts_graficos/script_graficos_v3.py` (eval tables),
    5) `scripts_graficos/script_target_distance_success.py` (success vs
       initial radius),
    6) optionally `inspect_episodes.py` and
       `scripts_graficos/script_plot_inspected_episodes.py` for detailed
       per-episode figures.
  - `watch_train.py` and `watch_arm.py` are interactive helpers for
    monitoring reward/success curves and 2D arm motion during/after
    training; they should not change training behavior.

- Extra plotting / analysis scripts (under `scripts_graficos/`):
  - `script_plot_train.py`: aggregates training curves across seeds and
    saves mean±std PDFs for reward, success and final distance.
  - `script_graficos_v3.py`: ingests all
    `results/eval_official_seed*_*.csv`, normalizes mode names and
    writes global and per-seed summary tables (CSV).
  - `script_target_distance_success.py`: studies success rate as a
    function of initial tip–target distance by binning `tip_init_dist`
    from eval CSVs and plotting pure vs pirl.
  - `inspect_episodes.py` + `script_plot_inspected_episodes.py`:
    generate detailed CSVs and figures for a few representative
    episodes (low/high energy successes and a high-energy failure).

## 10. Current tuning plan (2026-10): pure first, then pirl

The current high-level plan is to first make the `pure` mode robust,
then gradually add physics-informed terms for `pirl`.

- Phase 0 – Baseline setup
  - Use position-control env (`TwoLinkArmEnv`) with default
    `max_delta_theta=0.1`, `success_threshold=0.05`, `max-steps=200` in
    both train and eval.
  - PPO hyperparameters as in `train_rl.py` (PPO/MlpPolicy, lr≈3e-4,
    n_steps=2048, batch_size=256, etc.).

- Phase 1 – Coarse tuning of `pure` in a single seed (seed 0)
  - Vary only `lambda_a` (action penalty) and `success_bonus`, keeping
    `alpha_tau` irrelevant (only used in `pirl`).
  - Example configs (all with `steps=2e6`, `lr=3e-4`, `max-steps=200`):
    - `lambda_a=0.01`, `success_bonus=5.0` (baseline reference).
    - `lambda_a=0.005`, `success_bonus=5.0` (weaker action penalty).
    - `lambda_a=0.01`, `success_bonus≈8.0` (stronger terminal success
      bonus).
  - For each, run:
    1) `train_rl.py --mode pure --seed 0 ... --run-tag <tag>`,
    2) `eval_rl.py --mode pure --seed 0 --episodes 200 --max-steps 200
       --run-tag <tag> --out results/eval_pure_seed0_<tag>`.
  - Compare `success_rate` and `mean_final_dist` to pick a candidate
    region where success approaches or exceeds ~70%.

- Phase 2 – Refine around the best `pure` config
  - Once a promising pair (`lambda_a`, `success_bonus`) is found on seed
    0, refine with small adjustments (e.g. ±2–3 on success bonus or a
    slightly smaller `lambda_a`) until seed 0 reliably reaches ≥70%
    success.

- Phase 3 – Validate `pure` on multiple seeds
  - Fix the chosen `pure` hyperparameters and train/evaluate seeds
    0–4.
  - Use `run_plots_train.py` and `run_plots_eval_all.py` (or
    `run_plots_results_paper.py`) to inspect success rate across seeds.
  - Aim for mean success ≥70% with no catastrophic seeds.

  - Current baseline (2026-10): after coarse tuning on seed 0 and
    validation on seeds 0–4, the working `pure` configuration for
    experiments is:

    - `steps = 2_000_000`, `learning_rate = 3e-4`.
    - `lambda_a = 0.01`.
    - `success_bonus = 5.0`.
    - `max_delta_theta = 0.2` (larger action range than the default 0.1,
      improves convergence and final success).
    - `max_episode_steps = 200` in both train (TimeLimit) and eval.

    With this setup, seeds 0–4 for `pure` reach success rates in the
    ~0.7–0.98 range, with mean success ≈0.85 and reasonable final
    distances. New agents should treat this as the canonical `pure`
    baseline unless explicitly exploring alternatives.

- Phase 4 – Only then, gradually add physics terms to `pirl`
  - Start from the stable `pure` config and treat it as baseline.
  - For `pirl`, keep per-step reward close to `pure` and add physics
    terms carefully, e.g.:
    - very small `alpha_tau` (torque penalty) to avoid crushing success;
    - or terminal-only penalties based on episode energy,
      joint-limit/singularity proximity, or torque saturation rates.
  - Optionally, initialize `pirl` training from a converged `pure`
    policy (load pure model, swap env to `reward_mode="pirl"`, then
    continue training) so that physics terms refine an already-successful
    policy instead of competing with basic reaching behavior from
    scratch.

Agents extending the reward or scripts should keep this plan in mind:
do not change `pirl` physics terms until `pure` has been retuned and
validated, and prefer adding new optional parameters/flags over breaking
the existing CLI.
