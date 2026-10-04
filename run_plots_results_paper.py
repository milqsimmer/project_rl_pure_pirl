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
            "Pipeline de figuras para o capitulo de resultados: "
            "graficos de treino, comparacao de avaliacao, "
            "sucesso vs distancia inicial e episodios inspecionados."
        )
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="Seed inicial (inclusive) usada nos treinos/modelos.",
    )
    parser.add_argument(
        "--last-seed",
        type=int,
        default=4,
        help="Seed final (inclusive) usada nos treinos/modelos.",
    )
    parser.add_argument(
        "--episodes-per-inspect",
        type=int,
        default=20,
        help=(
            "Numero de episodios extras por (modo, seed) na etapa de "
            "inspect_episodes (usado apenas se --include-inspected)."
        ),
    )
    parser.add_argument(
        "--max-steps-inspect",
        type=int,
        default=200,
        help=(
            "Limite de passos por episodio na inspeção detalhada "
            "(apenas se --include-inspected).",
        ),
    )
    parser.add_argument(
        "--include-inspected",
        action="store_true",
        help=(
            "Se definido, roda inspect_episodes.py para seeds no intervalo "
            "e em seguida gera figuras com script_plot_inspected_episodes.py."
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

    # 1) Graficos de treino (reward + sucesso e sucesso por seed)
    run_cmd(
        [
            py,
            "run_plots_train.py",
            "--first-seed",
            str(args.first_seed),
            "--last-seed",
            str(args.last_seed),
        ]
    )

    # 2) Graficos de avaliacao (pure vs pirl por seed; + escalas opcionais)
    run_cmd(
        [
            py,
            "run_plots_eval_all.py",
            "--first-seed",
            str(args.first_seed),
            "--last-seed",
            str(args.last_seed),
            "--eval-out-prefix-template",
            "results/eval_official_seed{seed}",
            "--plot-prefix",
            "results/plot_eval",
            "--include-scales",
        ]
    )

    # 3) Graficos agregados de treino em PDF (script legado, mas util p/ paper)
    #    - fig_treino_reward.pdf
    #    - fig_treino_sucesso.pdf
    #    - fig_treino_distancia_final.pdf
    run_cmd([py, "scripts_graficos/script_plot_train.py"])

    # 4) Tabelas e resumos globais de avaliacao (script_graficos_v3)
    #    - tabelas/resumo_geral_todos_csvs.csv
    #    - tabelas/resumo_por_seed.csv
    run_cmd([py, "scripts_graficos/script_graficos_v3.py"])

    # 5) Sucesso em funcao da distancia inicial da ponta ao alvo
    #    - tabelas/target_distance_success.csv
    #    - figs_resultados/fig_target_distance_success.pdf
    run_cmd([py, "scripts_graficos/script_target_distance_success.py"])

    # 6) Episodios inspecionados (opcional)
    if args.include_inspected:
        seeds_str = ",".join(
            str(s) for s in range(args.first_seed, args.last_seed + 1)
        )

        # 6a) Gera CSVs de episodios representativos
        run_cmd(
            [
                py,
                "inspect_episodes.py",
                "--mode",
                "both",
                "--seeds",
                seeds_str,
                "--episodes-per-config",
                str(args.episodes_per_inspect),
                "--max-steps",
                str(args.max_steps_inspect),
            ]
        )

        # 6b) Gera figuras a partir dos CSVs de inspeção
        run_cmd([py, "scripts_graficos/script_plot_inspected_episodes.py"])


if __name__ == "__main__":
    main()
