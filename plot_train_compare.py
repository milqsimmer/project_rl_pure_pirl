import argparse
import os

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
            "Compara curvas de treino (reward e sucesso) entre diferentes seeds "
            "no mesmo gráfico, a partir dos monitor.csv do Stable-Baselines3."
        )
    )
    ap.add_argument("--mode", choices=["pure", "pirl", "both"], default="both")
    ap.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Lista de seeds separados por vírgula (ex.: 0,1,2).",
    )
    ap.add_argument(
        "--runs",
        type=str,
        default="",
        help=(
            "Lista de execuções no formato seed[:run_tag[:label]] separados por vírgula. "
            "Ex.: 0::seed0,1::seed1,0:lr1e4:seed0_lr1e4. Se fornecido, substitui --seeds/--run-tag."
        ),
    )
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
        "--out-prefix",
        type=str,
        default="results/train_compare",
        help=(
            "Prefixo para salvar figuras. Um arquivo por modo será gerado, "
            "ex.: results/train_compare_pure.png."
        ),
    )
    ap.add_argument(
        "--max-timesteps",
        type=int,
        default=2000000,
        help=(
            "Se definido, limita os gráficos aos timesteps de treino "
            "menores ou iguais a esse valor."
        ),
    )
    ap.add_argument(
        "--only-success-per-seed",
        action="store_true",
        help=(
            "Se definido, gera apenas o gráfico de sucesso por seed "
            "(um subplot por seed) e não gera o gráfico padrão."
        ),
    )
    return ap.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
    seeds: list[int] = []
    for part in seeds_str.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds


def _format_lr_from_run_tag(run_tag: str) -> str:
    """Converte run_tag do tipo 'lr3e4' para texto '3e-4'.

    Se não reconhecer o padrão, devolve o próprio run_tag.
    """
    # mapeia explicitamente alguns casos comuns
    mapping = {
        "lr1e4": "1e-4",
        "lr3e4": "3e-4",
        "lr5e4": "5e-4",
    }
    if run_tag in mapping:
        print(
            f"[plot_train_compare] run_tag '{run_tag}' mapeado para learning rate '{mapping[run_tag]}'"
        )
        return mapping[run_tag]

    if run_tag.startswith("lr"):
        body = run_tag[2:]
        if "e" in body:
            mantissa, exp = body.split("e", 1)
            if mantissa.replace(".", "", 1).isdigit() and exp.lstrip("-").isdigit():
                if not exp.startswith("-"):
                    exp = "-" + exp
                return f"{mantissa}e{exp}"
    return run_tag


def parse_runs(runs_str: str, fallback_run_tag: str = "") -> list[dict]:
    runs: list[dict] = []
    for chunk in runs_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if not parts[0]:
            continue
        seed = int(parts[0])
        run_tag = parts[1] if len(parts) >= 2 and parts[1] else fallback_run_tag
        if len(parts) >= 3 and parts[2]:
            label = parts[2]
        else:
            # legenda automaticamente baseada no learning rate
            if run_tag:
                lr_text = _format_lr_from_run_tag(run_tag)
                label = f"seed {seed} (lr={lr_text})"
            else:
                # valor default do train_rl.py
                label = f"seed {seed} (lr=3e-4)"
        runs.append({"seed": seed, "run_tag": run_tag, "label": label})
    return runs


def discover_runs_for_seeds(mode: str, seeds: list[int]) -> list[dict]:
    """Descobre automaticamente execuções (run_tag) para cada seed.

    Procura diretórios em runs_<mode>/seed_<seed>* que contenham monitor.csv
    e monta a lista de runs com labels baseados no run_tag (tipicamente
    representando o learning rate).
    """

    base_dir = f"runs_{mode}"
    if not os.path.isdir(base_dir):
        print(
            f"[plot_train_compare] Diretório de execuções não encontrado: {base_dir}. "
            "Nenhuma execução será descoberta."
        )
        return []

    runs: list[dict] = []
    try:
        entries = sorted(os.listdir(base_dir))
    except OSError as e:
        print(f"[plot_train_compare] Erro ao listar {base_dir}: {e}")
        return []

    for seed in seeds:
        prefix = f"seed_{seed}"
        for entry in entries:
            if not entry.startswith(prefix):
                continue

            run_dir = os.path.join(base_dir, entry)
            monitor_path = os.path.join(run_dir, "monitor.csv")
            if not os.path.isfile(monitor_path):
                continue

            suffix = entry[len(prefix) :]
            if suffix.startswith("_"):
                run_tag = suffix[1:]
            else:
                run_tag = suffix

            if run_tag:
                lr_text = _format_lr_from_run_tag(run_tag)
                label = f"seed {seed} (lr={lr_text})"
            else:
                label = f"seed {seed} (lr=3e-4)"

            runs.append({"seed": seed, "run_tag": run_tag, "label": label})

    return runs


def plot_mode(
    mode: str,
    runs: list[dict],
    window: int,
    out_prefix: str,
    max_timesteps=None,
) -> None:
    monitor_exists = False

    # garante diretório de saída
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, (ax_r, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_r.set_ylabel("Retorno (média móvel)")
    ax_r.set_title(f"Treino modo={mode} – comparação entre seeds")

    ax_s.set_xlabel("Timesteps de treino")
    ax_s.set_ylabel("Taxa de sucesso (média móvel)")
    ax_s.set_ylim(0.0, 1.0)

    any_success = False

    for run in runs:
        seed = run["seed"]
        run_tag = run["run_tag"]
        label = run["label"]

        monitor_path = get_monitor_path(mode, seed, run_tag)
        if not os.path.isfile(monitor_path):
            print(
                f"[plot_train_compare] Aviso: monitor.csv não encontrado para "
                f"mode={mode}, seed={seed}, run_tag='{run_tag}' "
                f"(caminho: {monitor_path}) – pulando."
            )
            continue

        try:
            # monitor.csv tem a primeira linha comentada com '#'
            df = pd.read_csv(monitor_path, skiprows=1)
        except Exception as e:
            print(
                f"[plot_train_compare] Erro lendo monitor.csv para "
                f"mode={mode}, seed={seed}: {e}"
            )
            continue

        if df.empty:
            print(
                f"[plot_train_compare] Arquivo vazio para mode={mode}, "
                f"seed={seed} – pulando."
            )
            continue

        monitor_exists = True

        timesteps = df["l"].cumsum()
        rewards = df["r"]

        if max_timesteps is not None:
            mask = timesteps <= max_timesteps
            if not mask.any():
                continue
            timesteps = timesteps[mask]
            rewards = rewards[mask]
        reward_ma = rewards.rolling(window=window, min_periods=1).mean()
        ax_r.plot(timesteps, reward_ma, label=label)

        if "is_success" in df.columns:
            success = df["is_success"]
            if max_timesteps is not None:
                success = success[mask]
            success_ma = success.rolling(window=window, min_periods=1).mean()
            ax_s.plot(timesteps, success_ma, label=label)
            any_success = True

    if not monitor_exists:
        print(
            f"[plot_train_compare] Nenhum monitor.csv encontrado para modo {mode}. "
            "Nada será salvo para este modo."
        )
        plt.close(fig)
        return

    ax_r.legend(loc="best")
    if any_success:
        ax_s.legend(loc="best")

    fig.tight_layout()

    out_path = f"{out_prefix}_{mode}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_train_compare] Figura salva em: {out_path}")


def plot_mode_success_per_seed(
    mode: str,
    runs: list[dict],
    window: int,
    out_prefix: str,
    max_timesteps=None,
) -> None:
    """Plota apenas a taxa de sucesso, separando um gráfico por seed.

    Cada subplot corresponde a uma seed e, dentro dele, são mostradas as
    curvas das diferentes execuções (tipicamente com LRs diferentes).
    """

    # garante diretório de saída
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # agrupa execuções por seed
    seeds: list[int] = []
    for run in runs:
        s = run["seed"]
        if s not in seeds:
            seeds.append(s)

    if not seeds:
        print(
            f"[plot_train_compare] Nenhuma seed fornecida para modo {mode} "
            "em plot_mode_success_per_seed."
        )
        return

    n_seeds = len(seeds)

    fig, axes = plt.subplots(n_seeds, 1, sharex=True, figsize=(8, 2.5 * n_seeds))
    if n_seeds == 1:
        axes = [axes]

    any_curve = False

    for ax, seed in zip(axes, seeds):
        ax.set_ylabel("Taxa de sucesso")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"modo={mode} – seed {seed}")

        for run in runs:
            if run["seed"] != seed:
                continue

            run_tag = run["run_tag"]
            label = run["label"]

            monitor_path = get_monitor_path(mode, seed, run_tag)
            if not os.path.isfile(monitor_path):
                print(
                    f"[plot_train_compare] Aviso: monitor.csv não encontrado para "
                    f"mode={mode}, seed={seed}, run_tag='{run_tag}' "
                    f"(caminho: {monitor_path}) – pulando."
                )
                continue

            try:
                df = pd.read_csv(monitor_path, skiprows=1)
            except Exception as e:
                print(
                    f"[plot_train_compare] Erro lendo monitor.csv para "
                    f"mode={mode}, seed={seed}: {e}"
                )
                continue

            if df.empty:
                print(
                    f"[plot_train_compare] Arquivo vazio para mode={mode}, "
                    f"seed={seed} – pulando."
                )
                continue

            if "is_success" not in df.columns:
                print(
                    f"[plot_train_compare] Coluna 'is_success' não encontrada em "
                    f"mode={mode}, seed={seed} – pulando."
                )
                continue

            timesteps = df["l"].cumsum()

            if max_timesteps is not None:
                mask = timesteps <= max_timesteps
                if not mask.any():
                    continue
                timesteps = timesteps[mask]

            success = df["is_success"]
            if max_timesteps is not None:
                success = success[mask]
            success_ma = success.rolling(window=window, min_periods=1).mean()
            ax.plot(timesteps, success_ma, label=label)
            any_curve = True

        ax.legend(loc="best")

    axes[-1].set_xlabel("Timesteps de treino")

    if not any_curve:
        print(
            f"[plot_train_compare] Nenhuma curva de sucesso encontrada para modo {mode} "
            "em plot_mode_success_per_seed. Figura não será salva."
        )
        plt.close(fig)
        return

    fig.tight_layout()

    out_path = f"{out_prefix}_{mode}_success_per_seed.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_train_compare] Figura (sucesso por seed) salva em: {out_path}")


def main() -> None:
    args = parse_args()
    runs_arg = args.runs.strip()

    if runs_arg:
        parsed_runs = parse_runs(runs_arg)
        base_seeds = sorted({r["seed"] for r in parsed_runs})
    else:
        base_seeds = parse_seeds(args.seeds)

    if args.mode == "both":
        modes = ["pure", "pirl"]
    else:
        modes = [args.mode]

    for mode in modes:
        if runs_arg:
            runs = parsed_runs
        else:
            runs = []
            for s in base_seeds:
                run_tag = args.run_tag
                if run_tag:
                    lr_text = _format_lr_from_run_tag(run_tag)
                    label = f"seed {s} (lr={lr_text})"
                else:
                    label = f"seed {s} (lr=3e-4)"
                runs.append({"seed": s, "run_tag": run_tag, "label": label})

            # modo automático: descobrir todos os run_tags por seed quando
            # o usuário pediu apenas sucesso por seed e não especificou
            # run-tag nem lista de runs.
            if args.only_success_per_seed and not args.run_tag:
                auto_runs = discover_runs_for_seeds(mode, base_seeds)
                if auto_runs:
                    runs = auto_runs

        if args.only_success_per_seed:
            # apenas sucesso, um subplot por seed
            plot_mode_success_per_seed(
                mode,
                runs,
                args.window,
                args.out_prefix,
                max_timesteps=args.max_timesteps,
            )
        else:
            # gráfico padrão (reward + sucesso no mesmo painel)
            plot_mode(
                mode,
                runs,
                args.window,
                args.out_prefix,
                max_timesteps=args.max_timesteps,
            )

            # gráfico adicional: apenas sucesso, um subplot por seed
            plot_mode_success_per_seed(
                mode,
                runs,
                args.window,
                args.out_prefix,
                max_timesteps=args.max_timesteps,
            )


if __name__ == "__main__":
    main()
