import argparse
import json
import os
import random
import sys
import numpy as np
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from two_link_arm_env import TwoLinkArmEnv


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_run_dir(mode: str, seed: int, run_tag: str = "") -> str:
    suffix = f"_" + run_tag if run_tag else ""
    return os.path.join("runs", f"runs_{mode}", f"seed_{seed}{suffix}")


def get_model_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), f"ppo_model_{seed}.zip")


def get_monitor_path(mode: str, seed: int, run_tag: str = "") -> str:
    return os.path.join(get_run_dir(mode, seed, run_tag), "monitor.csv")


def save_train_command_log(args: argparse.Namespace, run_dir: str) -> None:
    """Salva, na pasta de treino, o comando e os parametros usados."""

    log_path = os.path.join(run_dir, "train_cmd.txt")
    try:
        cmdline = " ".join([sys.executable] + sys.argv[1:])
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("command: " + cmdline + "\n")
            f.write(
                "args_json: "
                + json.dumps(vars(args), ensure_ascii=False, indent=2)
                + "\n"
            )
    except Exception as e:
        print(f"[train_rl] Aviso: nao foi possivel salvar log de comando: {e}")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["pure", "pirl"], default="pure")
parser.add_argument("--steps", type=int, default=300_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--run-tag",
    type=str,
    default="",
    help="Sufixo opcional para distinguir execucoes com mesma seed.",
)
parser.add_argument(
    "--lambda-a",
    type=float,
    default=0.03,
    dest="lambda_a",
    help="Peso da penalidade de acao na recompensa.",
)
parser.add_argument(
    "--alpha-tau",
    type=float,
    default=0.001,
    help="Peso da penalidade de torque na recompensa (modo pirl).",
)
parser.add_argument(
    "--alpha-tau-mode",
    choices=["per_step", "terminal"],
    default="per_step",
    dest="alpha_tau_mode",
    help=(
        "Como alpha_tau age no modo pirl: per_step (padrao) = penaliza "
        "alpha_tau*(|tau1|+|tau2|) a cada passo; terminal = penaliza alpha_tau*torque "
        "medio so no passo de sucesso (runs *_term)."
    ),
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=3e-4,
    dest="learning_rate",
    help="Taxa de aprendizado do otimizador PPO.",
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Se definido, abre a GUI do PyBullet durante o treino.",
)
parser.add_argument(
    "--step-sleep",
    type=float,
    default=0.0,
    help=(
        "Pausa em segundos apos cada passo de simulacao quando --render esta ativo "
        "(para desacelerar a animacao)."
    ),
)
parser.add_argument(
    "--target-seed",
    type=int,
    default=None,
    help=(
        "Seed usada apenas para amostrar os alvos do ambiente. "
        "Permite alinhar a sequencia de alvos entre pure/pirl."
    ),
)
parser.add_argument(
    "--max-delta-theta",
    type=float,
    default=0.1,
    dest="max_delta_theta",
    help=(
        "Modulo maximo da variacao de angulo por passo (acao) em radianos. "
        "Default=0.1 (~5.7 graus)."
    ),
)
parser.add_argument(
    "--success-bonus",
    type=float,
    default=0.0,
    dest="success_bonus",
    help=(
        "Bonus adicional somado ao reward no passo em que o alvo e alcancado "
        "(dist < success_threshold). Default=0.0 (sem bonus)."
    ),
)
parser.add_argument(
    "--beta-omega",
    type=float,
    default=0.0,
    dest="beta_omega",
    help=(
        "Peso da penalizacao por passo de velocidade articular acima do envelope "
        "(80%% de 5 rad/s) no modo pirl. Default=0.0 (sem efeito)."
    ),
)
parser.add_argument(
    "--beta-tau",
    type=float,
    default=0.0,
    dest="beta_tau",
    help=(
        "Peso da penalizacao por passo de torque acima do envelope (80%% de 20 N.m) "
        "no modo pirl. Default=0.0 (sem efeito)."
    ),
)
parser.add_argument(
    "--beta-theta",
    type=float,
    default=0.0,
    dest="beta_theta",
    help=(
        "Peso da penalizacao por passo de |theta| acima de 2.512 rad (80%% do limite 3.14) "
        "no modo pirl. Default=0.0 (sem efeito)."
    ),
)
parser.add_argument(
    "--normalize-actions",
    action="store_true",
    dest="normalize_actions",
    help=(
        "Espaco de acao da politica em [-1, 1] (o ambiente escala por "
        "max_delta_theta). Evita a saturacao da media da politica no limite da acao. "
        "Use a mesma flag na avaliacao."
    ),
)
args = parser.parse_args()

set_global_seed(args.seed)

target_seed_effective = args.target_seed if args.target_seed is not None else args.seed

run_dir = get_run_dir(args.mode, args.seed, args.run_tag)
os.makedirs(run_dir, exist_ok=True)

monitor_path = get_monitor_path(args.mode, args.seed, args.run_tag)
model_path = get_model_path(args.mode, args.seed, args.run_tag)

save_train_command_log(args, run_dir)

env = TwoLinkArmEnv(
    render=args.render,
    reward_mode=args.mode,
    lambda_a=args.lambda_a,
    alpha_tau=args.alpha_tau,
    step_sleep=args.step_sleep,
    max_delta_theta=args.max_delta_theta,
    target_seed=target_seed_effective,
    success_bonus=args.success_bonus,
    beta_omega=args.beta_omega,
    beta_tau=args.beta_tau,
    beta_theta=args.beta_theta,
    normalize_actions=args.normalize_actions,
    alpha_tau_mode=args.alpha_tau_mode,
)
env = TimeLimit(env, max_episode_steps=200)

# IMPORTANTE:
# info_keywords registra colunas extras no monitor.csv no fim de cada episódio
env = Monitor(
    env,
    filename=monitor_path,
    info_keywords=(
        "is_success",
        "final_distance",
        "tau_sum_total",
        "episode_energy",
        "episode_mean_tau_sum",
    ),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=args.learning_rate,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

model.learn(total_timesteps=args.steps)

model.save(model_path)
env.close()

print(f"[OK] Treino {args.mode} finalizado.")
print(f"[OK] Monitor salvo em: {monitor_path}")
print(f"[OK] Modelo salvo em: {model_path}")
