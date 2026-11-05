"""Gráficos de treino a partir dos monitor.csv do Stable-Baselines3.

Executar a partir da raiz do projeto. Lê runs_<mode>/seed_<seed>_<tag>/monitor.csv
(convenção de train_rl.py) e gera, em --out-dir (default figs/figs_treino):

  agregado (--figures aggregate)  PDF, média ± desvio entre seeds, pure x pirl
      fig_treino_reward.pdf, fig_treino_sucesso.pdf, fig_treino_distancia_final.pdf
  compare  (--figures compare)    PNG, uma curva por seed, um arquivo por modo
      train_compare_<mode>.png  (retorno + sucesso)
  por seed (--figures per-seed)   PNG, sucesso com um subplot por seed
      train_compare_<mode>_success_per_seed.png

Exemplos:
  python scripts/graficos/plot_train.py --seeds 1-10
  python scripts/graficos/plot_train.py --seeds 0-4 --figures aggregate
  python scripts/graficos/plot_train.py --seeds 0-4 --pure-tag md02 --pirl-tag md02_at3e4
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "pure": "RL puro",
    "pirl": "PIRL (PI-reward)",
}

# run_tag das runs oficiais (config md02; pirl com alpha_tau=2e-4)
DEFAULT_TAGS = {"pure": "md02", "pirl": "md02_at2e4"}

FIGURES = ("aggregate", "compare", "per-seed")


# ========= I/O =========
def get_monitor_path(mode: str, seed: int, run_tag: str = "") -> str:
    """Replica a convenção de pastas de train_rl.py (runs_pure/seed_0_md02)."""
    suffix = f"_{run_tag}" if run_tag else ""
    return os.path.join(f"runs_{mode}", f"seed_{seed}{suffix}", "monitor.csv")


def read_monitor(path: str) -> pd.DataFrame | None:
    """Lê um monitor.csv (primeira linha é comentário '#'). None se ausente/vazio."""
    if not os.path.isfile(path):
        print(f"[plot_train] Aviso: não encontrado: {path} – pulando.")
        return None
    try:
        df = pd.read_csv(path, skiprows=1)
    except Exception as e:  # noqa: BLE001
        print(f"[plot_train] Erro lendo {path}: {e} – pulando.")
        return None
    if df.empty:
        print(f"[plot_train] Arquivo vazio: {path} – pulando.")
        return None
    # r = retorno, l = comprimento, + info_keywords (is_success, final_distance, ...)
    df["timesteps"] = df["l"].cumsum()
    return df


def load_mode(mode: str, seeds: list[int], run_tag: str) -> pd.DataFrame:
    """Concatena os monitors das seeds de um modo, com coluna 'seed'."""
    dfs = []
    for seed in seeds:
        df = read_monitor(get_monitor_path(mode, seed, run_tag))
        if df is None:
            continue
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def add_moving_averages(df: pd.DataFrame, window: int) -> None:
    for src, dst in (
        ("r", "reward_ma"),
        ("is_success", "success_ma"),
        ("final_distance", "final_distance_ma"),
    ):
        if src in df.columns:
            df[dst] = df.groupby("seed")[src].transform(
                lambda s: moving_average(s, window)
            )


def cut_timesteps(df: pd.DataFrame, max_timesteps: int | None) -> pd.DataFrame:
    if max_timesteps is None or df.empty:
        return df
    return df[df["timesteps"] <= max_timesteps]


# ========= AGREGADO (média ± desvio entre seeds) =========
def aggregate_curves(df: pd.DataFrame, y_col: str, max_points: int = 300):
    """Interpola as seeds numa grade comum de timesteps. Retorna x, média, desvio."""
    g = df.groupby("seed")["timesteps"]
    x_min, x_max = g.min().max(), g.max().min()
    if x_max <= x_min:
        raise ValueError(f"Faixa inválida para agregação: {x_min} -> {x_max}")

    x_grid = np.linspace(x_min, x_max, max_points)
    curves = []
    for _, d in df.groupby("seed"):
        d = d.sort_values("timesteps")
        x_unique, idx = np.unique(d["timesteps"].values, return_index=True)
        curves.append(np.interp(x_grid, x_unique, d[y_col].values[idx]))
    curves = np.array(curves)
    ddof = 1 if len(curves) > 1 else 0
    return x_grid, curves.mean(axis=0), curves.std(axis=0, ddof=ddof)


def plot_aggregate(dfs: dict[str, pd.DataFrame], window: int, out_dir: str) -> None:
    specs = [
        ("reward_ma", f"Retorno por episódio (média móvel, janela={window})",
         "fig_treino_reward.pdf", None),
        ("success_ma", f"Taxa de sucesso (média móvel, janela={window})",
         "fig_treino_sucesso.pdf", (0.0, 1.0)),
        ("final_distance_ma", f"Distância final (média móvel, janela={window})",
         "fig_treino_distancia_final.pdf", None),
    ]
    for y_col, ylabel, fname, ylim in specs:
        fig, ax = plt.subplots()
        plotted = False
        for mode, df in dfs.items():
            if df.empty or y_col not in df.columns:
                continue
            x, y, s = aggregate_curves(df, y_col)
            n = df["seed"].nunique()
            ax.plot(x, y, label=f"{LABELS[mode]} (n={n})")
            ax.fill_between(x, y - s, y + s, alpha=0.2)
            plotted = True
        if not plotted:
            print(f"[plot_train] Nada para plotar em {fname}.")
            plt.close(fig)
            continue
        ax.set_xlabel("Timesteps de treino")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_train] Figura salva em: {path}")


# ========= COMPARE (uma curva por seed) =========
def plot_compare(mode: str, df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        print(f"[plot_train] Nenhum dado para modo {mode}; compare não gerado.")
        return
    fig, (ax_r, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax_r.set_ylabel("Retorno (média móvel)")
    ax_r.set_title(f"Treino modo={mode} – comparação entre seeds")
    ax_s.set_xlabel("Timesteps de treino")
    ax_s.set_ylabel("Taxa de sucesso (média móvel)")
    ax_s.set_ylim(0.0, 1.0)

    any_success = False
    for seed, d in df.groupby("seed"):
        ax_r.plot(d["timesteps"], d["reward_ma"], label=f"seed {seed}")
        if "success_ma" in d.columns:
            ax_s.plot(d["timesteps"], d["success_ma"], label=f"seed {seed}")
            any_success = True

    ax_r.legend(loc="best")
    if any_success:
        ax_s.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_compare_{mode}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[plot_train] Figura salva em: {path}")


def plot_per_seed(mode: str, df: pd.DataFrame, out_dir: str) -> None:
    if df.empty or "success_ma" not in df.columns:
        print(f"[plot_train] Sem curvas de sucesso para modo {mode}; per-seed não gerado.")
        return
    seeds = sorted(df["seed"].unique())
    fig, axes = plt.subplots(
        len(seeds), 1, sharex=True, figsize=(8, 2.5 * len(seeds)), squeeze=False
    )
    for ax, seed in zip(axes[:, 0], seeds):
        d = df[df["seed"] == seed]
        ax.plot(d["timesteps"], d["success_ma"])
        ax.set_ylabel("Taxa de sucesso")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"modo={mode} – seed {seed}")
    axes[-1, 0].set_xlabel("Timesteps de treino")
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_compare_{mode}_success_per_seed.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[plot_train] Figura (sucesso por seed) salva em: {path}")


# ========= CLI =========
def parse_seeds(text: str) -> list[int]:
    """Aceita '0,1,2', '1-10' ou combinações ('0-2,5')."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Gráficos de treino (agregado pure x pirl, comparação e sucesso por seed)."
    )
    ap.add_argument("--mode", choices=["pure", "pirl", "both"], default="both")
    ap.add_argument("--seeds", default="0-4", help="Ex.: 0,1,2 ou 1-10.")
    ap.add_argument(
        "--run-tag", default="",
        help="run_tag comum aos dois modos (sobrescreve os defaults; vazio = defaults).",
    )
    ap.add_argument("--pure-tag", default=None,
                    help=f"run_tag do pure (default: {DEFAULT_TAGS['pure']}).")
    ap.add_argument("--pirl-tag", default=None,
                    help=f"run_tag do pirl (default: {DEFAULT_TAGS['pirl']}).")
    ap.add_argument("--window", type=int, default=1000,
                    help="Janela da média móvel, em episódios.")
    ap.add_argument("--max-timesteps", type=int, default=2_000_000,
                    help="Corta as curvas neste número de timesteps.")
    ap.add_argument("--figures", default=",".join(FIGURES),
                    help=f"Subconjunto de {', '.join(FIGURES)} (separado por vírgula).")
    ap.add_argument("--out-dir", default="figs/figs_treino")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    figures = {f.strip() for f in args.figures.split(",") if f.strip()}
    unknown = figures - set(FIGURES)
    if unknown:
        raise SystemExit(f"--figures inválido: {sorted(unknown)}; use {FIGURES}")

    modes = ["pure", "pirl"] if args.mode == "both" else [args.mode]
    tags = dict(DEFAULT_TAGS)
    if args.run_tag:
        tags = {m: args.run_tag for m in tags}
    if args.pure_tag is not None:
        tags["pure"] = args.pure_tag
    if args.pirl_tag is not None:
        tags["pirl"] = args.pirl_tag

    os.makedirs(args.out_dir, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for mode in modes:
        df = cut_timesteps(load_mode(mode, seeds, tags[mode]), args.max_timesteps)
        if not df.empty:
            add_moving_averages(df, args.window)
        dfs[mode] = df

    if "aggregate" in figures:
        plot_aggregate(dfs, args.window, args.out_dir)
    for mode in modes:
        if "compare" in figures:
            plot_compare(mode, dfs[mode], args.out_dir)
        if "per-seed" in figures:
            plot_per_seed(mode, dfs[mode], args.out_dir)


if __name__ == "__main__":
    main()
