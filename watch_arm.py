import argparse
import os
import time

import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO

from two_link_arm_env import TwoLinkArmEnv
from plot_arm_from_env import plot_arm


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    """Replica a convenção de pastas de train_rl.py."""

    suffix = f"_{run_tag}" if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}")


def get_model_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), f"ppo_model_{seed}.zip")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Visualiza em 2D o movimento do braco para um modelo PPO treinado "
            "(pure ou pirl), usando o mesmo layout do plot_arm_from_env."
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
        "--max-steps",
        type=int,
        default=200,
        help="Número máximo de passos por episódio (TimeLimit).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Intervalo em segundos entre frames da animação.",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=0,
        help=(
            "Número de episódios a rodar (0 = loop infinito até fechar a janela)."
        ),
    )
    ap.add_argument(
        "--pause-between-episodes",
        action="store_true",
        help=(
            "Se definido, aguarda Enter no terminal entre o fim de um episódio "
            "e o início do próximo."
        ),
    )
    return ap.parse_args()


def unwrap_env(env):
    """Desenvolve wrappers simples (TimeLimit, Monitor) até chegar no TwoLinkArmEnv."""

    base = env
    # gym 0.21: wrappers costumam expor .env
    while hasattr(base, "env"):
        base = base.env
    return base


def main() -> None:
    args = parse_args()

    model_path = get_model_path(args.mode, args.seed, args.run_tag)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    print(f"[watch_arm] Carregando modelo de: {model_path}")

    env = TwoLinkArmEnv(render=False, reward_mode=args.mode)
    env = TimeLimit(env, max_episode_steps=args.max_steps)

    model = PPO.load(model_path, env=env)

    base_env = unwrap_env(env)

    l1 = getattr(base_env, "l1", 0.5)
    l2 = getattr(base_env, "l2", 0.5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    episode_idx = 0
    total_episodes = args.episodes if args.episodes > 0 else None

    try:
        while True:
            if total_episodes is not None and episode_idx >= total_episodes:
                break

            obs = env.reset()
            done = False
            episode_idx += 1
            step_idx = 0

            print(f"[watch_arm] Episódio {episode_idx}")

            while not done:
                # obtém ângulos e alvo atuais do ambiente base
                theta1 = float(getattr(base_env, "theta1", 0.0))
                theta2 = float(getattr(base_env, "theta2", 0.0))
                target_pos = getattr(base_env, "target_pos", None)

                if target_pos is not None:
                    target = (float(target_pos[0]), float(target_pos[1]))
                else:
                    target = None

                ax.clear()
                plot_arm(ax, theta1, theta2, l1, l2, target=target)
                ax.set_title(
                    f"modo={args.mode}, seed={args.seed}, ep={episode_idx}, step={step_idx}",
                    fontsize=10,
                )

                fig.canvas.draw()
                fig.canvas.flush_events()

                time.sleep(args.sleep)

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                step_idx += 1

                if not plt.fignum_exists(fig.number):
                    # janela foi fechada pelo usuário
                    return

            # fim do episódio atual
            if args.pause_between_episodes:
                try:
                    input(
                        "[watch_arm] Episódio finalizado. Pressione Enter para o próximo "
                        "(Ctrl+C para sair)..."
                    )
                except (KeyboardInterrupt, EOFError):
                    return

    finally:
        env.close()
        plt.ioff()
        plt.close(fig)
        print("[watch_arm] Encerrado.")


if __name__ == "__main__":
    main()
