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
            "Gera os graficos de treino a partir dos monitor.csv: "
            "(1) reward + sucesso e (2) taxa de sucesso por seed, "
            "para modos pure e pirl."
        )
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="Seed inicial (inclusive) usada nos treinos.",
    )
    parser.add_argument(
        "--last-seed",
        type=int,
        default=4,
        help="Seed final (inclusive) usada nos treinos.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Comando do interpretador Python (padrao: o atual).",
    )

    args = parser.parse_args()

    if args.first_seed > args.last_seed:
        raise SystemExit("first-seed deve ser <= last-seed")

    py = args.python_exe
    seeds_str = ",".join(str(s) for s in range(args.first_seed, args.last_seed + 1))

    # 1) Graficos completos (reward + sucesso) para pure e pirl
    cmd_full = [
        py,
        "plot_train_compare.py",
        "--mode",
        "both",
        "--seeds",
        seeds_str,
    ]
    run_cmd(cmd_full)

    # 2) Graficos de sucesso por seed (um subplot por seed)
    cmd_success = [
        py,
        "plot_train_compare.py",
        "--mode",
        "both",
        "--seeds",
        seeds_str,
        "--only-success-per-seed",
    ]
    run_cmd(cmd_success)


if __name__ == "__main__":
    main()
