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
            "Roda avaliacoes (pure e/ou pirl) para um intervalo de seeds de modelo, "
            "permitindo usar um intervalo possivelmente diferente de target_seeds "
            "(sementes dos alvos do ambiente)."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["pure", "pirl", "both"],
        default="both",
        help="Quais modos avaliar (pure, pirl ou both).",
    )
    parser.add_argument(
        "--model-first-seed",
        type=int,
        default=0,
        help="Seed inicial dos modelos (inclusive).",
    )
    parser.add_argument(
        "--model-last-seed",
        type=int,
        default=4,
        help="Seed final dos modelos (inclusive).",
    )
    parser.add_argument(
        "--target-first-seed",
        type=int,
        default=None,
        help=(
            "Seed inicial para os alvos do ambiente (target_seed). Se nao for "
            "informado, usa a mesma dos modelos."
        ),
    )
    parser.add_argument(
        "--target-last-seed",
        type=int,
        default=None,
        help=(
            "Seed final para os alvos do ambiente (target_seed). Se nao for "
            "informado, usa a mesma dos modelos."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Numero de episodios por avaliacao.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Limite de passos por episodio (TimeLimit).",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Comando do interpretador Python (padrao: o atual).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Sufixo opcional usado no diretorio de treino (seed_X_<tag>).",
    )

    args = parser.parse_args()

    if args.model_first_seed > args.model_last_seed:
        raise SystemExit("model-first-seed deve ser <= model-last-seed")

    n_models = args.model_last_seed - args.model_first_seed + 1

    if args.target_first_seed is None and args.target_last_seed is None:
        use_same_target = True
    elif args.target_first_seed is not None and args.target_last_seed is not None:
        use_same_target = False
        n_targets = args.target_last_seed - args.target_first_seed + 1
        if n_targets != n_models:
            raise SystemExit(
                "Intervalo de target_seeds deve ter o mesmo tamanho do intervalo "
                "de model_seeds."
            )
    else:
        raise SystemExit(
            "Use ambos --target-first-seed e --target-last-seed ou nenhum deles."
        )

    py = args.python_exe

    for idx, model_seed in enumerate(
        range(args.model_first_seed, args.model_last_seed + 1)
    ):
        if use_same_target:
            target_seed = model_seed
        else:
            target_seed = args.target_first_seed + idx

        print("\n" + "=" * 80)
        print(
            f"Avaliando modelos seed={model_seed} com target_seed={target_seed} "
            f"(mode={args.mode})"
        )
        print("=" * 80)

        out_prefix = (
            f"results/eval_model{model_seed}_target{target_seed}_mode{args.mode}"
        )

        cmd_eval = [
            py,
            "eval_rl.py",
            "--mode",
            args.mode,
            "--seed",
            str(model_seed),
            "--episodes",
            str(args.episodes),
            "--max-steps",
            str(args.max_steps),
            "--target-seed",
            str(target_seed),
            "--out",
            out_prefix,
        ]
        if args.run_tag:
            cmd_eval.extend(["--run-tag", args.run_tag])

        run_cmd(cmd_eval)


if __name__ == "__main__":
    main()
