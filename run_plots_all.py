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
            "Gera os graficos principais de treino (reward + sucesso), "
            "taxa de sucesso por seed e graficos de escalas (analyze_scales) "
            "para pure e pirl, a partir dos artefatos gerados pelos treinos."
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
        "--max-timesteps",
        type=int,
        default=2_000_000,
        help=(
            "Valor de corte opcional em timesteps para a versao '_max2000steps' "
            "dos graficos. Default=2e6.",
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

    # 3) Versoes cortadas em max_timesteps, com sufixo '_max2000steps'
    out_prefix = f"results/train_compare_max2000steps"

    cmd_full_cut = [
        py,
        "plot_train_compare.py",
        "--mode",
        "both",
        "--seeds",
        seeds_str,
        "--max-timesteps",
        str(args.max_timesteps),
        "--out-prefix",
        out_prefix,
    ]
    run_cmd(cmd_full_cut)

    cmd_success_cut = [
        py,
        "plot_train_compare.py",
        "--mode",
        "both",
        "--seeds",
        seeds_str,
        "--only-success-per-seed",
        "--max-timesteps",
        str(args.max_timesteps),
        "--out-prefix",
        out_prefix,
    ]
    run_cmd(cmd_success_cut)

    # 4) Graficos de escalas (distancia, torque, energia) para cada seed
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

    # 5) Figura esquematica do braco a partir do env (opcional, roda uma vez)
    try:
        run_cmd([py, "plot_arm_from_env.py"])
    except SystemExit:
        # se faltar URDF ou algo semelhante, nao interrompe o restante
        print("[run_plots_all] plot_arm_from_env.py falhou; seguindo em frente.")


if __name__ == "__main__":
    main()
