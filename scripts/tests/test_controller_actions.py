"""Teste do controlador sob sequencias de acoes tipicas de PPO (sem RL).

Requisitos: numpy, pybullet, matplotlib (graficos opcionais).
Uso: python scripts/tests/test_controller_actions.py --urdf two_link_arm.urdf

Compara frequencias de decisao 30 Hz e 60 Hz; fisica sempre 240 Hz.
O incremento angular e aplicado a uma referencia persistente, NAO a posicao medida.
Um episodio de 2 segundos por perfil e frequencia; as duas juntas sao acionadas.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import pybullet as p

DT = 1.0 / 240.0
PHYSICAL_STEPS = 480
JOINTS = (0, 1)
FORCE_MAX = 20.0
VELOCITY_MAX = 5.0
Q_MIN, Q_MAX = -3.14, 3.14
PROFILES = ('aleatorio_01', 'alternado_01', 'pequeno_002', 'grande_01')


def action_for(profile, index, rng):
    if profile == 'aleatorio_01':
        return rng.uniform(-0.1, 0.1, 2)
    if profile == 'alternado_01':
        return np.array([0.1, -0.1]) * (1 if index % 2 == 0 else -1)
    if profile == 'pequeno_002':
        return np.array([0.02, -0.02])  # rampa lenta constante, com sinais opostos
    if profile == 'grande_01':
        return np.array([0.1, -0.1])  # rampa mais rapida, com sinais opostos
    raise ValueError(profile)


def run_case(robot, client, hz, profile, seed):
    substeps = 240 // hz
    rng = np.random.default_rng(seed)
    for j in JOINTS:
        p.resetJointState(robot, j, targetValue=0.0, targetVelocity=0.0,
                          physicsClientId=client)
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                targetPosition=0.0, force=FORCE_MAX,
                                maxVelocity=VELOCITY_MAX, physicsClientId=client)
    for _ in range(24):
        p.stepSimulation(physicsClientId=client)

    ref = np.zeros(2, dtype=float)
    samples = []
    for k in range(PHYSICAL_STEPS):
        if k % substeps == 0:
            action = action_for(profile, k // substeps, rng)
            ref = np.clip(ref + action, Q_MIN, Q_MAX)
            for j in JOINTS:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                        targetPosition=float(ref[j]),
                                        force=FORCE_MAX, maxVelocity=VELOCITY_MAX,
                                        physicsClientId=client)
        p.stepSimulation(physicsClientId=client)
        state = p.getJointStates(robot, list(JOINTS), physicsClientId=client)
        row = {'profile': profile, 'decision_hz': hz, 'physics_step': k,
               'time_s': (k + 1) * DT,
               'ref1_rad': float(ref[0]), 'ref2_rad': float(ref[1])}
        for j in JOINTS:
            row[f'q{j+1}_rad'] = float(state[j][0])
            row[f'omega{j+1}_rad_s'] = float(state[j][1])
            row[f'tau{j+1}_Nm'] = float(state[j][3])
        samples.append(row)
    return samples


def summarize(rows):
    q = np.array([[r['q1_rad'], r['q2_rad']] for r in rows])
    ref = np.array([[r['ref1_rad'], r['ref2_rad']] for r in rows])
    omega = np.array([[r['omega1_rad_s'], r['omega2_rad_s']] for r in rows])
    tau = np.array([[r['tau1_Nm'], r['tau2_Nm']] for r in rows])
    absomega, abstau = np.abs(omega), np.abs(tau)
    over5 = absomega > VELOCITY_MAX + 1e-3  # tolerancia para classificacao
    above4 = absomega > 4.0
    above16 = abstau > 16.0
    near20 = abstau >= 19.8
    result = {'profile': rows[0]['profile'], 'decision_hz': rows[0]['decision_hz'],
              'duration_s': PHYSICAL_STEPS * DT}
    for j in JOINTS:
        suffix = str(j + 1)
        result.update({
            f'peak_velocity_j{suffix}_rad_s': float(absomega[:, j].max()),
            f'peak_torque_j{suffix}_Nm': float(abstau[:, j].max()),
            f'velocity_gt5_j{suffix}_pct': float(100 * over5[:, j].mean()),
            f'velocity_gt4_j{suffix}_pct': float(100 * above4[:, j].mean()),
            f'torque_gt16_j{suffix}_pct': float(100 * above16[:, j].mean()),
            f'torque_near20_j{suffix}_pct': float(100 * near20[:, j].mean()),
            f'mean_tracking_error_j{suffix}_rad': float(np.mean(np.abs(ref[:, j]-q[:, j]))),
            f'final_tracking_error_j{suffix}_rad': float(abs(ref[-1, j]-q[-1, j])),
            f'abs_torque_integral_j{suffix}_Nms': float(np.sum(abstau[:, j])*DT),
            f'abs_mech_work_j{suffix}_J': float(np.sum(abstau[:, j]*absomega[:, j])*DT),
        })
    result['any_velocity_gt5_pct'] = float(100 * over5.any(axis=1).mean())
    result['any_torque_near20_pct'] = float(100 * near20.any(axis=1).mean())
    return result


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plot(rows, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t = np.array([r['time_s'] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j in JOINTS:
        i = j + 1
        axes[0].plot(t, [r[f'ref{i}_rad'] for r in rows], '--', label=f'Ref J{i}')
        axes[0].plot(t, [r[f'q{i}_rad'] for r in rows], label=f'Pos J{i}')
        axes[1].plot(t, [r[f'omega{i}_rad_s'] for r in rows], label=f'Vel J{i}')
        axes[2].plot(t, [r[f'tau{i}_Nm'] for r in rows], label=f'Torque J{i}')
    for sgn in [-1, 1]:
        axes[1].axhline(sgn*VELOCITY_MAX, color='gray', linestyle=':', linewidth=1)
        axes[2].axhline(sgn*FORCE_MAX, color='gray', linestyle=':', linewidth=1)
    for ax, ylabel in zip(axes, ['Posicao (rad)', 'Velocidade (rad/s)', 'Torque (N.m)']):
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Tempo simulado (s)')
    fig.suptitle(f"{rows[0]['profile']} | decisao {rows[0]['decision_hz']} Hz | fisica 240 Hz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', default=str(Path(__file__).resolve().parents[2] / 'two_link_arm.urdf'))
    parser.add_argument('--out-dir', default='resultados_controle_acoes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    urdf = Path(args.urdf).resolve()
    if not urdf.exists():
        parser.error(f'URDF inexistente: {urdf}')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    all_samples = []
    all_summary = []
    plots = 0
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(DT, physicsClientId=client)
        robot = p.loadURDF(str(urdf), useFixedBase=True,
                           flags=p.URDF_USE_INERTIA_FROM_FILE,
                           physicsClientId=client)
        if p.getNumJoints(robot, physicsClientId=client) != 2:
            raise RuntimeError('Esperadas exatamente duas juntas.')
        for hz in (30, 60):
            for profile in PROFILES:
                rows = run_case(robot, client, hz, profile, args.seed)
                all_samples.extend(rows)
                result = summarize(rows)
                all_summary.append(result)
                plots += make_plot(rows, out / f'{profile}_{hz}hz.png')
    finally:
        p.disconnect(physicsClientId=client)
    save_csv(out / 'amostras.csv', all_samples)
    save_csv(out / 'resumo.csv', all_summary)
    print('\nACOES VARIAVEIS: 2 s por perfil, fisica 240 Hz; force 20 N.m; maxVelocity 5 rad/s')
    print('Perfil          Hz  | pico vel J1/J2 (rad/s) | torque J1/J2 (N.m) | >5 (alguma junta) | ~20 (alguma junta) | erro medio J1/J2 (rad)')
    for r in all_summary:
        print(f"{r['profile']:<16}{r['decision_hz']:>3} | "
              f"{r['peak_velocity_j1_rad_s']:6.3f}/{r['peak_velocity_j2_rad_s']:6.3f}         | "
              f"{r['peak_torque_j1_Nm']:5.2f}/{r['peak_torque_j2_Nm']:5.2f}          | "
              f"{r['any_velocity_gt5_pct']:6.2f}%             | "
              f"{r['any_torque_near20_pct']:6.2f}%              | "
              f"{r['mean_tracking_error_j1_rad']:.3f}/{r['mean_tracking_error_j2_rad']:.3f}")
    print(f'\nDados em {out.resolve()} ({plots} graficos, se matplotlib disponivel).')
    print('Nota: referencias nao sao trajetorias equivalentes entre 30 e 60 Hz; comparar estabilidade, nao desempenho de alcance.')


if __name__ == '__main__':
    main()
