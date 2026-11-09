"""Treinamento piloto PPO para TwoLinkArmEnv (RL puro).

Uso: python train_rl.py --seed 0 --steps 300000 --run-tag pilot
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from two_link_arm_env import TwoLinkArmEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=300_000)
    ap.add_argument('--run-tag', default='pilot')
    ap.add_argument('--learning-rate', type=float, default=3e-4)
    ap.add_argument('--lambda-action', type=float, default=0.01)
    ap.add_argument('--success-bonus', type=float, default=5.0)
    ap.add_argument('--max-delta-theta', type=float, default=0.1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tensorboard', action='store_true')
    args = ap.parse_args()
    if args.steps <= 0 or args.max_delta_theta <= 0:
        ap.error('--steps e --max-delta-theta devem ser positivos')
    # Protege contra sobrescrita silenciosa de treinos existentes.
    tag = f'_{args.run_tag}' if args.run_tag else ''
    run_dir = Path('runs') / 'pure' / f'seed_{args.seed}{tag}'
    model_path = run_dir / 'ppo_model.zip'
    if model_path.exists():
        ap.error(f'Modelo já existe: {model_path} (use outro --run-tag)')
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'args': vars(args),
        'versions': {'python': sys.version, 'gymnasium': gymnasium.__version__,
                     'sb3': stable_baselines3.__version__, 'numpy': np.__version__,
                     'torch': torch.__version__},
        'algorithm': 'PPO',
        'policy_kwargs': {'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'activation_fn': 'Tanh'},
        'ppo_kwargs': {'n_steps': 2048, 'batch_size': 256, 'n_epochs': 10,
                       'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2},
        'physics': {'physics_hz': 240, 'decision_hz': 30, 'episode_seconds': 2.0,
                    'force_limit': 20.0, 'velocity_limit': 5.0},
    }
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    (run_dir / 'command.txt').write_text(' '.join([sys.executable] + sys.argv[1:]), encoding='utf8')

    env = TwoLinkArmEnv(lambda_action=args.lambda_action,
                          success_bonus=args.success_bonus,
                          max_delta_theta=args.max_delta_theta)
    env.action_space.seed(args.seed)
    # Monitor coleta métricas entregues pelo próprio ambiente no fim do episódio.
    monitor = Monitor(env, filename=str(run_dir / 'monitor.csv'),
                      info_keywords=('is_success', 'distance', 'elapsed_s',
                                     'torque_abs_integral', 'torque_sq_integral',
                                     'energy_abs', 'torque_envelope_rate',
                                     'velocity_envelope_rate', 'position_envelope_rate',
                                     'torque_saturation_rate', 'safe_episode', 'safe_success'))
    try:
        monitor.reset(seed=args.seed)
        model = PPO('MlpPolicy', monitor, seed=args.seed,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                       activation_fn=torch.nn.Tanh),
                    learning_rate=args.learning_rate, n_steps=2048,
                    batch_size=256, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2,
                    tensorboard_log=str(Path('runs') / 'tensorboard') if args.tensorboard else None,
                    device=args.device, verbose=1)
        model.learn(total_timesteps=args.steps,
                    tb_log_name=f'pure_s{args.seed}{tag}')
        model.save(str(model_path))
    finally:
        monitor.close()
    print(f'\nTreinamento concluído. Modelo: {model_path}')
    print(f'Monitor: {run_dir / "monitor.csv"}')


if __name__ == '__main__':
    main()
