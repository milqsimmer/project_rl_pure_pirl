import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    print("\n=== Executando:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Roda treinamentos (pure e pirl) para um intervalo de seeds, "
            "usando o mesmo valor como seed do PPO e target_seed do ambiente, "
            "com hiperparametros padrao alinhados a configuracao atual "
            "(lambda_a=0.01, alpha_tau=2e-4, steps=2e6, success_bonus opcional)."
        )
    )
    parser.add_argument(
        "--first-seed", type=int, default=0, help="Seed inicial (inclusive)."
    )
    parser.add_argument(
        "--last-seed", type=int, default=4, help="Seed final (inclusive)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2_000_000,
        help="Numero de passos de treino por seed (default: 2e6).",
    )
    parser.add_argument(
        "--lambda-a",
        type=float,
        default=0.01,
        dest="lambda_a",
        help="Peso lambda_a da penalidade de acao (default: 0.01).",
    )
    parser.add_argument(
        "--alpha-tau",
        type=float,
        default=0.0002,
        help=(
            "Peso alpha_tau da penalidade de torque no modo pirl (default: 2e-4), "
            "alinhado com a configuracao atual.",
        ),
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Comando do interpretador Python (padrao: o atual).",
    )
    parser.add_argument(
        "--success-bonus",
        type=float,
        default=5.0,
        help=(
            "Bonus adicional somado ao reward no passo de sucesso (default: 5.0). "
            "Passado para train_rl.py como --success-bonus."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help=(
            "Sufixo opcional para distinguir execucoes com mesma seed "
            "(passado para train_rl.py como --run-tag)."
        ),
    )
    parser.add_argument(
        "--skip-pure",
        action="store_true",
        help="Pula os treinos do modo pure.",
    )
    parser.add_argument(
        "--skip-pirl",
        action="store_true",
        help="Pula os treinos do modo pirl.",
    )

    args = parser.parse_args()

    if args.first_seed > args.last_seed:
        raise SystemExit("first-seed deve ser <= last-seed")

    py = args.python_exe

    for seed in range(args.first_seed, args.last_seed + 1):
        print("\n" + "=" * 80)
        print(f"Seed {seed} (seed PPO e target_seed do ambiente)")
        print("=" * 80)

        # modo pure
        if not args.skip_pure:
            cmd_pure = [
                py,
                "train_rl.py",
                "--mode",
                "pure",
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--lambda-a",
                str(args.lambda_a),
                "--target-seed",
                str(seed),
                "--success-bonus",
                str(args.success_bonus),
            ]
            if args.run_tag:
                cmd_pure.extend(["--run-tag", args.run_tag])
            run_cmd(cmd_pure)

        # modo pirl
        if not args.skip_pirl:
            cmd_pirl = [
                py,
                "train_rl.py",
                "--mode",
                "pirl",
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--lambda-a",
                str(args.lambda_a),
                "--alpha-tau",
                str(args.alpha_tau),
                "--target-seed",
                str(seed),
                "--success-bonus",
                str(args.success_bonus),
            ]
            if args.run_tag:
                cmd_pirl.extend(["--run-tag", args.run_tag])
            run_cmd(cmd_pirl)


if __name__ == "__main__":
    main()
