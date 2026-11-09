"""Diagnostico do controlador POSITION_CONTROL do braco 2-DoF no PyBullet.

Executa degrau, rampa e reversao, para cada junta isoladamente, sem PPO.
Mantem force=20 e os ganhos padrao do POSITION_CONTROL.
Compara omissao de maxVelocity (controle original) com maxVelocity=5 rad/s.
Salva amostras por passo fisico, resumo e graficos.

Uso:
    python scripts/tests/test_controller_velocity.py --urdf two_link_arm.urdf
    python scripts/tests/test_controller_velocity.py --urdf two_link_arm.urdf --out-dir resultados_controle_velocidade
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pybullet as p

DT = 1.0 / 240.0
FORCE = 20.0  # N.m (limite passado explicitamente ao motor)
URDF_SPEED_REFERENCE = 5.0  # rad/s (declarado no URDF; nao presumir enforcement)
TORQUE_ENVELOPE = 16.0  # N.m; criterio exploratorio de 80%
SPEED_ENVELOPE = 4.0  # rad/s; criterio exploratorio de 80%
STEPS = 360  # 1.5 s para cada perfil
JOINTS = (0, 1)


def reference(kind, t):
    """Retorna alvo em radianos para a junta que esta sendo testada."""
    if kind == "degrau":
        return 0.30 if t >= 0.10 else 0.0
    if kind == "rampa":
        return float(np.clip((t - 0.10) / 0.75, 0.0, 1.0) * 0.60)
    if kind == "reversao":
        if t < 0.10:
            return 0.0
        if t < 0.65:
            return 0.40
        return -0.40
    raise ValueError(f"Perfil desconhecido: {kind}")


def reset_and_hold(robot, client, max_velocity):
    for joint in JOINTS:
        p.resetJointState(robot, joint, targetValue=0.0,
                          targetVelocity=0.0, physicsClientId=client)
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL,
                                targetPosition=0.0, force=FORCE,
                                **({"maxVelocity": max_velocity} if max_velocity is not None else {}),
                                physicsClientId=client)
    # Elimina efeitos transitorios de reset antes da coleta.
    for _ in range(24):
        p.stepSimulation(physicsClientId=client)


def simulate(robot, client, joint, kind, max_velocity, mode):
    reset_and_hold(robot, client, max_velocity)
    samples = []
    for k in range(STEPS):
        t = k * DT
        target = reference(kind, t)
        # Ambas as juntas sao comandadas; a outra recebe referencia zero.
        for j in JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=target if j == joint else 0.0,
                                    force=FORCE,
                                    **({"maxVelocity": max_velocity} if max_velocity is not None else {}),
                                    physicsClientId=client)
        p.stepSimulation(physicsClientId=client)
        state = p.getJointStates(robot, list(JOINTS), physicsClientId=client)
        row = {"mode": mode, "profile": kind, "tested_joint": joint + 1,
               "step": k, "time_s": (k + 1) * DT, "target_rad": target}
        for j in JOINTS:
            row[f"q{j+1}_rad"] = float(state[j][0])
            row[f"omega{j+1}_rad_s"] = float(state[j][1])
            row[f"tau{j+1}_Nm"] = float(state[j][3])
        samples.append(row)
    return samples


def summarize(rows):
    joint = rows[0]["tested_joint"]
    omega = np.array([r[f"omega{joint}_rad_s"] for r in rows])
    tau = np.array([r[f"tau{joint}_Nm"] for r in rows])
    q = np.array([r[f"q{joint}_rad"] for r in rows])
    target = np.array([r["target_rad"] for r in rows])
    return {
        "mode": rows[0]["mode"], "profile": rows[0]["profile"], "tested_joint": joint,
        "peak_speed_rad_s": float(np.max(np.abs(omega))),
        "peak_torque_Nm": float(np.max(np.abs(tau))),
        "speed_gt4_pct": float(100 * np.mean(np.abs(omega) > SPEED_ENVELOPE)),
        "speed_gt5_pct": float(100 * np.mean(np.abs(omega) > URDF_SPEED_REFERENCE)),
        "torque_gt16_pct": float(100 * np.mean(np.abs(tau) > TORQUE_ENVELOPE)),
        "torque_near20_pct": float(100 * np.mean(np.abs(tau) >= 19.8)),
        "final_tracking_error_rad": float(abs(q[-1] - target[-1])),
        "integral_abs_tau_Nms": float(np.sum(np.abs(tau)) * DT),
        "integral_abs_power_J": float(np.sum(np.abs(tau * omega)) * DT),
    }


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_profile(rows, out_path):
    import matplotlib.pyplot as plt
    joint = rows[0]["tested_joint"]
    name = rows[0]["profile"]
    t = np.array([r["time_s"] for r in rows])
    target = np.array([r["target_rad"] for r in rows])
    q = np.array([r[f"q{joint}_rad"] for r in rows])
    w = np.array([r[f"omega{joint}_rad_s"] for r in rows])
    tau = np.array([r[f"tau{joint}_Nm"] for r in rows])

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, target, "--", label="Referencia")
    ax[0].plot(t, q, label="Posicao medida")
    ax[0].set_ylabel("Posicao (rad)")
    ax[0].legend()
    ax[1].plot(t, w, label="Velocidade medida")
    ax[1].axhline(URDF_SPEED_REFERENCE, linestyle="--", label="Referencia +5 rad/s")
    ax[1].axhline(-URDF_SPEED_REFERENCE, linestyle="--")
    ax[1].set_ylabel("Velocidade (rad/s)")
    ax[1].legend()
    ax[2].plot(t, tau, label="Torque do motor")
    ax[2].axhline(FORCE, linestyle="--", label="Limite +20 N.m")
    ax[2].axhline(-FORCE, linestyle="--")
    ax[2].set_ylabel("Torque (N.m)")
    ax[2].set_xlabel("Tempo simulado (s)")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{rows[0]['mode']} - teste {name} - junta {joint}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path(__file__).resolve().parents[2] / "two_link_arm.urdf")
    parser.add_argument("--out-dir", type=Path, default=Path("resultados_controle_velocidade"))
    args = parser.parse_args()

    urdf = args.urdf.resolve()
    if not urdf.is_file():
        parser.error(f"URDF nao encontrado: {urdf}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(DT, physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        robot = p.loadURDF(str(urdf), basePosition=[0, 0, 0], useFixedBase=True,
                           flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
        assert p.getNumJoints(robot, physicsClientId=client) == 2, "Esperadas duas juntas"

        all_rows, summaries = [], []
        # Baseline omite maxVelocity exatamente como no script anterior.
        for mode, vmax in (("original", None), ("limitado_5", 5.0)):
            for kind in ("degrau", "rampa", "reversao"):
                for joint in JOINTS:
                    rows = simulate(robot, client, joint, kind, vmax, mode)
                    all_rows.extend(rows)
                    summaries.append(summarize(rows))
                    plot_profile(rows, args.out_dir / f"{mode}_{kind}_junta{joint+1}.png")
        save_csv(args.out_dir / "amostras.csv", all_rows)
        save_csv(args.out_dir / "resumo.csv", summaries)

        print("\nCOMPARACAO DO CONTROLADOR (1,5 s por perfil):")
        print("Modo           Perfil    Junta | Pico vel. | Pico torque | >5 rad/s | ~20 N.m | Erro final")
        for s in summaries:
            print(f"{s['mode']:<15}{s['profile']:<11}{s['tested_joint']:>2}   |"
                  f" {s['peak_speed_rad_s']:>7.2f}   |"
                  f" {s['peak_torque_Nm']:>8.2f}    |"
                  f" {s['speed_gt5_pct']:>6.1f}%  |"
                  f" {s['torque_near20_pct']:>6.1f}% |"
                  f" {s['final_tracking_error_rad']:.4f} rad")
        print(f"\nArquivos salvos em: {args.out_dir.resolve()}")
        print("Nota: maxVelocity limita a referencia do motor, nao necessariamente a velocidade fisica absoluta.")
    finally:
        p.disconnect(physicsClientId=client)


if __name__ == "__main__":
    main()
