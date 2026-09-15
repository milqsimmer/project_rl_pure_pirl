import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    """Replica a convenção de pastas de train_rl.py.

    Ex.: runs_pure/seed_0_s300k
    """

    suffix = f"_{run_tag}" if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}")


def get_monitor_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), "monitor.csv")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Visualiza em tempo real as curvas de treino (reward e sucesso) "
            "a partir do monitor.csv do Stable-Baselines3."
        )
    )
    ap.add_argument("--mode", choices=["pure", "pirl"], default="pure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Sufixo opcional do diretório de treino (seed_X_<tag>).",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Tamanho da janela da média móvel em episódios.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Intervalo em segundos entre atualizações do gráfico.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    monitor_path = get_monitor_path(args.mode, args.seed, args.run_tag)
    print(f"[watch_train] Monitor alvo: {monitor_path}")

    plt.ion()
    fig, (ax_r, ax_s) = plt.subplots(2, 1, sharex=True)

    (line_r,) = ax_r.plot([], [], label="reward (média móvel)")
    ax_r.set_ylabel("Retorno")
    ax_r.legend(loc="best")

    (line_s,) = ax_s.plot([], [], label="sucesso (média móvel)")
    ax_s.set_xlabel("Timesteps de treino")
    ax_s.set_ylabel("Taxa de sucesso")
    ax_s.set_ylim(0.0, 1.0)
    ax_s.legend(loc="best")

    fig.tight_layout()

    first_wait_msg = True

    while True:
        if not os.path.isfile(monitor_path):
            if first_wait_msg:
                print("[watch_train] Aguardando monitor.csv ser criado...")
                first_wait_msg = False
            plt.pause(args.interval)
            continue

        try:
            # monitor.csv tem a primeira linha comentada com '#'
            df = pd.read_csv(monitor_path, skiprows=1)
        except Exception as e:
            print(f"[watch_train] Erro lendo monitor.csv: {e}")
            plt.pause(args.interval)
            continue

        if df.empty:
            plt.pause(args.interval)
            continue

        # eixo x em timesteps acumulados (como em script_plot_train.py)
        # coluna "l" = comprimento do episódio
        timesteps = df["l"].cumsum()

        # média móvel do reward por episódio
        rewards = df["r"]
        reward_ma = rewards.rolling(window=args.window, min_periods=1).mean()

        # média móvel de sucesso se a coluna existir
        if "is_success" in df.columns:
            success = df["is_success"]
            success_ma = success.rolling(window=args.window, min_periods=1).mean()
        else:
            success_ma = None

        # atualiza curvas
        line_r.set_data(timesteps, reward_ma)

        # define limites de eixo de forma razoável
        x_min = float(timesteps.min())
        x_max = float(timesteps.max())
        if x_min == x_max:
            x_min = 0.0
            x_max = x_max + 1.0
        ax_r.set_xlim(x_min, x_max)

        r_min = float(reward_ma.min())
        r_max = float(reward_ma.max())
        if r_min == r_max:
            r_min -= 1.0
            r_max += 1.0
        margin = 0.1 * max(1.0, abs(r_max - r_min))
        ax_r.set_ylim(r_min - margin, r_max + margin)

        if success_ma is not None:
            line_s.set_data(timesteps, success_ma)
            ax_s.set_xlim(x_min, x_max)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
