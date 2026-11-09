"""Avaliação determinística pareada para políticas treinadas no ambiente atual.

Não reaproveitar com pesos legados: espaço de observação e dinâmica mudaram.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from two_link_arm_env import TwoLinkArmEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0, help='Seed do modelo treinado')
    ap.add_argument('--run-tag', default='pilot')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--target-seed', type=int, default=10000, help='Mesma seed de alvos para todos os modelos')
    ap.add_argument('--model-path', default=None)
    ap.add_argument('--out', default=None, help='Caminho do CSV, sem sufixo automático')
    ap.add_argument('--print-episodes', action='store_true')
    args = ap.parse_args()
    if args.episodes <= 0:
        ap.error('--episodes deve ser positivo')
    tag = f'_{args.run_tag}' if args.run_tag else ''
    model_path = Path(args.model_path) if args.model_path else Path('runs') / 'pure' / f'seed_{args.seed}{tag}' / 'ppo_model.zip'
    config_path = model_path.with_name('config.json')
    if not model_path.exists():
        ap.error(f'Modelo não encontrado: {model_path}')
    if not config_path.exists():
        ap.error(f'Configuração do treino não encontrada: {config_path}. Informe o modelo dentro da pasta original.')
    config = json.loads(config_path.read_text(encoding='utf8'))
    cargs = config['args']
    physics = config.get('physics', {})
    env = TwoLinkArmEnv(lambda_action=float(cargs['lambda_action']),
                          success_bonus=float(cargs['success_bonus']),
                          max_delta_theta=float(cargs['max_delta_theta']),
                          physics_hz=int(physics.get('physics_hz', 240)),
                          decision_hz=int(physics.get('decision_hz', 30)),
                          episode_seconds=float(physics.get('episode_seconds', 2.0)),
                          force_limit=float(physics.get('force_limit', 20.0)),
                          velocity_limit=float(physics.get('velocity_limit', 5.0)))
    model = PPO.load(str(model_path), device='cpu')
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        env.close()
        raise ValueError('Espaços de observação ou ação não correspondem ao ambiente atual')
    rows = []
    try:
        for ep in range(args.episodes):
            # Seed aplicada SOMENTE no primeiro reset: mesma sequência de alvos
            # para agentes distintos com o mesmo --target-seed.
            obs, reset_info = env.reset(seed=args.target_seed if ep == 0 else None)
            target = np.array(reset_info['target_xy'], dtype=float)
            initial_dist = float(reset_info['distance'])
            initial_r = float(np.linalg.norm(target))
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            physical_steps = int(info['physics_steps'])
            elapsed_s = float(info['elapsed_s'])
            row = {
                'mode': 'pure', 'seed': args.seed, 'target_seed': args.target_seed,
                'episode': ep + 1, 'success': int(info['is_success']),
                'terminated': int(terminated), 'truncated': int(truncated),
                'final_distance': float(info['distance']),
                'return': total_reward, 'length': int(info['decision_steps']),
                'physics_steps': physical_steps, 'elapsed_s': elapsed_s,
                'target_x': float(target[0]), 'target_y': float(target[1]),
                'target_radius': initial_r, 'tip_init_dist': initial_dist,
                'torque_abs_integral': float(info['torque_abs_integral']),
                'torque_sq_integral': float(info['torque_sq_integral']),
                'energy_abs': float(info['energy_abs']),
                'peak_tau': float(np.max(np.abs(info['torque_peak']))),
                'peak_omega': float(np.max(np.abs(info['velocity_peak']))),
                'torque_envelope_rate': float(info['torque_envelope_rate']),
                'velocity_envelope_rate': float(info['velocity_envelope_rate']),
                'position_envelope_rate': float(info['position_envelope_rate']),
                'tau_sat_rate': float(info['torque_saturation_rate']),
                'safe_episode': int(info['safe_episode']),
                'safe_success': int(info['safe_success']),
                'action_clip_decisions': int(info['action_clip_decisions']),
                'reference_clip_decisions': int(info['reference_clip_decisions']),
                # Compatibilidade de nomes com CSVs antigos, mas a implementação atual integra cada
                # passo físico. Estes aliases são derivados de valores físicos reais.
                'energy': float(info['energy_abs']),
                'mean_tau_sum': (float(info['torque_abs_integral']) / elapsed_s
                                 if elapsed_s > 0 else float('nan')),
                'tau_sum_total': (float(info['torque_abs_integral']) / env.dt
                                  if env.dt > 0 else float('nan')),
            }
            rows.append(row)
            if args.print_episodes:
                print(f"ep={ep+1:03d} sucesso={row['success']} distância={row['final_distance']:.4f} "
                      f"passos={row['length']:02d} seguro={row['safe_episode']}")
    finally:
        env.close()
    out = Path(args.out) if args.out else Path('results') / f'eval_pure_seed{args.seed}{tag}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = ('success', 'final_distance', 'length', 'elapsed_s', 'torque_abs_integral',
               'torque_sq_integral', 'energy_abs', 'peak_tau', 'peak_omega',
               'torque_envelope_rate', 'velocity_envelope_rate', 'position_envelope_rate',
               'tau_sat_rate', 'safe_episode', 'safe_success')
    summary = {k: {'mean': float(np.mean([r[k] for r in rows])),
                   'std_between_episodes': float(np.std([r[k] for r in rows], ddof=1)) if len(rows)>1 else 0.0}
               for k in numeric}
    summary['metadata'] = {'model_path': str(model_path), 'train_seed': args.seed,
                           'target_seed': args.target_seed, 'episodes': args.episodes,
                           'notes': 'Variabilidade entre episódios, não entre seeds de treinamento.'}
    summ_path = out.with_name(out.stem + '_summary.json')
    summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(f"\nSucesso: {summary['success']['mean']:.1%} | "
          f"distância final média: {summary['final_distance']['mean']:.4f} m | "
          f"sucesso seguro: {summary['safe_success']['mean']:.1%}")
    print(f'CSV: {out}\nResumo: {summ_path}')


if __name__ == '__main__':
    main()
