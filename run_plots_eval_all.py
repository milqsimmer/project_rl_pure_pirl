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
            "Gera graficos a partir dos resultados de avaliacao: "
            "comparacao pure vs pirl por seed (sucesso, distancia final, "
            "torque, energia) e, opcionalmente, graficos de escalas "
            "(analyze_scales)."
        )
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="Seed inicial (inclusive) dos modelos avaliados.",
    )
    parser.add_argument(
        "--last-seed",
        type=int,
        default=4,
        help="Seed final (inclusive) dos modelos avaliados.",
    )
    parser.add_argument(
        "--eval-out-prefix-template",
        type=str,
        default="results/eval_official_seed{seed}",
        help=(
            "Template do prefixo usado em eval_rl.py --out. "
            "Ex.: results/eval_official_seed{seed}."
        ),
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="results/plot_eval",
        help="Prefixo para salvar as figuras de comparacao de eval.",
    )
    parser.add_argument(
        "--include-scales",
        action="store_true",
        help=(
            "Se definido, tambem roda analyze_scales.py para cada seed "
            "(pure e pirl, policy=trained)."
        ),
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

    # 1) Graficos de comparacao pure vs pirl por seed a partir dos JSON de eval
    cmd_eval_plots = [
        py,
        "plot_eval_compare.py",
        "--first-seed",
        str(args.first_seed),
        "--last-seed",
        str(args.last_seed),
        "--out-prefix-template",
        args.eval_out_prefix_template,
        "--plot-prefix",
        args.plot_prefix,
    ]
    run_cmd(cmd_eval_plots)

    # 2) (Opcional) Graficos de escalas para cada seed
    if args.include_scales:
        for seed in range(args.first_seed, args.last_seed + 1):
            out_prefix_scales = f"results/analyze_scales_seed{seed}"
            cmd_scales = [
                py,
                "analyze_scales.py",
                "--mode",
                "both",
                "--policy",
                "trained",
                "--seed",
                str(seed),
                "--out-prefix",
                out_prefix_scales,
            ]
            run_cmd(cmd_scales)


if __name__ == "__main__":
    main()
