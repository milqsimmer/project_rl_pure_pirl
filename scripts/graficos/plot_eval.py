"""Gráficos e tabelas de avaliação (pure x pirl) a partir dos CSVs de eval_rl.py.

Executar a partir da raiz do projeto. Lê, por episódio:

    <results-dir>/eval_model<seed>_target<seed>_mode{pure,pirl}_<seed>.csv
    (e o formato legado <results-dir>/eval_official_seed<seed>_<seed>.csv)
    com --results-dir (default results/md02).

Tabelas (--tables-dir, default tabelas/):
    resumo_geral_todos_csvs.csv   métricas agregadas por modo (todos os episódios)
    resumo_por_seed.csv           métricas médias por (seed, modo)
    target_distance_success.csv   sucesso por faixa de distância inicial ponta-alvo
    limites_movimento.csv         limites articulares, velocidade, torque e jerk
                                  (média ± dp entre seeds; requer colunas do eval_rl.py atual)

Figuras (--out-dir, default figs/figs_resultados/), escolhidas com --figures:
    boxplot         fig1_boxplot_combined.pdf          boxplot por seed (sucesso, distância)
    summary         fig3_resumo_barras.pdf             barras média±dp entre seeds (sucesso, dist, torque, energia)
    tradeoff        fig4_tradeoff_energia_sucesso.pdf  energia x sucesso, um ponto por seed
    seed-grid       fig_per_seed_grid.pdf              barras pure x pirl por seed, 4 métricas
    target-distance fig_target_distance_success.pdf    sucesso x distância inicial ponta-alvo
    motion          fig_limites_movimento.pdf          saturação de torque, velocidade/limite articular, jerk

Exemplos:
    python scripts/graficos/plot_eval.py                       # seeds 1-10, tudo
    python scripts/graficos/plot_eval.py --first-seed 0 --last-seed 4 --figures summary,tradeoff
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = ["pure", "pirl"]
LABELS = {"pure": "RL puro", "pirl": "PIRL (PI-reward)"}

CSV_PATTERNS = [
    "eval_official_seed*_*.csv",
    "eval_model*_target*_mode*_*.csv",
]

REQUIRED_COLS = ["mode", "episode", "success", "final_distance", "return", "length"]
NUM_COLS = ["episode", "success", "final_distance", "return", "length", "seed"]

# bins de distância inicial ponta-alvo (m) para a figura target-distance
DIST_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
DIST_BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]

# métricas por seed usadas nas barras: (coluna em seed_summary, rótulo, ylim)
M_SUCCESS = ("success_rate", "Taxa de sucesso", (0.0, 1.0))
M_DIST = ("final_distance", "Distância final (m)", None)
M_TAU = ("tau_effort", "Esforço médio em torque", None)
M_ENERGY = ("energy", "Energia média por episódio", None)


# ========= LEITURA E NORMALIZAÇÃO =========
def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.read_csv(path, sep=";")


def extract_seed_from_filename(path: str) -> int:
    """eval_official_seed3_3.csv -> 3; eval_model1_target1_modepure_1.csv -> 1."""
    name = os.path.basename(path)
    for pattern in (
        r"eval_official_seed(\d+)_\d+\.csv$",
        r"eval_model(\d+)_target\d+_mode[^_]+_\d+\.csv$",
        r"eval_results_(\d+)\.csv$",  # legado
    ):
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    raise ValueError(f"Não foi possível extrair a seed do nome do arquivo: {name}")


def normalize_mode(value) -> str:
    """Normaliza nomes de modo para 'pure' ou 'pirl' (outros: devolve o texto)."""
    if pd.isna(value):
        return ""
    v = str(value).strip().lower()
    if v in {"pure", "rl", "baseline", "rl puro", "rl_puro", "pure_rl"}:
        return "pure"
    if v in {"pirl", "pi-reward", "pi_reward", "physics-informed",
             "physics_informed", "physics informed"}:
        return "pirl"
    if "pure" in v or "baseline" in v:
        return "pure"
    if "pirl" in v or "pi-reward" in v or "physics" in v:
        return "pirl"
    return v


def load_eval_data(
    first_seed: int,
    last_seed: int,
    results_dir: str,
    pure_dir: str | None = None,
    pirl_dir: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Lê os CSVs de eval das seeds [first_seed, last_seed] em um único DataFrame.

    Por padrão pure e pirl vêm de results_dir. Com pure_dir e/ou pirl_dir, as
    linhas de cada modo são lidas da pasta indicada (para comparar, por exemplo,
    o pure de uma pasta com o pirl de outra).
    """
    sources = [(pure_dir or results_dir, "pure"), (pirl_dir or results_dir, "pirl")]
    if (pure_dir or results_dir) == (pirl_dir or results_dir):
        sources = [(results_dir if not pure_dir else pure_dir, None)]

    dfs, used, globs = [], [], []
    for folder, only_mode in sources:
        patterns = [os.path.join(folder, pat) for pat in CSV_PATTERNS]
        globs.extend(patterns)
        for path in sorted({p for pat in patterns for p in glob.glob(pat)}):
            seed = extract_seed_from_filename(path)
            if not first_seed <= seed <= last_seed:
                continue
            df = read_csv_auto(path)
            if only_mode is not None:
                df = df[df["mode"].apply(normalize_mode) == only_mode]
                if df.empty:
                    continue
            df["seed"] = seed
            df["source_file"] = os.path.basename(path)
            dfs.append(df)
            used.append(path)
    if not dfs:
        raise SystemExit(
            f"Nenhum CSV de eval para as seeds {first_seed}-{last_seed} "
            f"({', '.join(globs)})."
        )

    df = pd.concat(dfs, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas faltando nos CSVs: {missing}; esperado {REQUIRED_COLS}")

    df["mode_norm"] = df["mode"].apply(normalize_mode)
    df = df[df["mode_norm"].isin(MODES)].copy()
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["mode_norm", "seed", "success", "final_distance", "return", "length"])
    return df.copy(), used


# ========= TABELAS =========
def _mean_std(series: pd.Series) -> tuple[float, float]:
    return series.mean(), series.std(ddof=1)


def build_general_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas agregadas por modo, sobre todos os episódios."""
    rows = []
    for mode in MODES:
        d = df[df["mode_norm"] == mode]
        row = {"mode": LABELS[mode], "success_rate_mean": d["success"].mean()}
        for key, col in (("final_distance", "final_distance"), ("length", "length"),
                         ("return", "return")):
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(d[col])
        for key, col in (("tau_effort", "mean_tau_sum"), ("energy", "energy")):
            row[f"{key}_mean"], row[f"{key}_std"] = (
                _mean_std(d[col]) if col in d.columns else (np.nan, np.nan)
            )
        row["n_episodes"] = len(d)
        rows.append(row)
    return pd.DataFrame(rows)


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas médias por (seed, modo)."""
    agg = {
        "success_rate": ("success", "mean"),
        "final_distance": ("final_distance", "mean"),
        "episode_length": ("length", "mean"),
        "episode_return": ("return", "mean"),
        "n_episodes": ("episode", "count"),
    }
    if "mean_tau_sum" in df.columns:
        agg["tau_effort"] = ("mean_tau_sum", "mean")
    if "energy" in df.columns:
        agg["energy"] = ("energy", "mean")
    out = (
        df.groupby(["seed", "mode_norm"], as_index=False)
        .agg(**agg)
        .sort_values(["seed", "mode_norm"])
    )
    out["mode_label"] = out["mode_norm"].map(LABELS)
    return out


def build_target_distance_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Sucesso por faixa de distância inicial da ponta ao alvo."""
    if "tip_init_dist" not in df.columns:
        print("[plot_eval] Coluna 'tip_init_dist' ausente; target-distance ignorado.")
        return None
    d = df.copy()
    d["tip_init_dist"] = pd.to_numeric(d["tip_init_dist"], errors="coerce")
    d = d.dropna(subset=["tip_init_dist"])
    d["radius_bin"] = pd.cut(
        d["tip_init_dist"], bins=DIST_BINS, labels=DIST_BIN_LABELS, include_lowest=True
    )
    d = d.dropna(subset=["radius_bin"])  # alvos fora dos bins
    grp = (
        d.groupby(["mode_norm", "radius_bin"], observed=True)
        .agg(
            success_rate=("success", "mean"),
            final_distance=("final_distance", "mean"),
            n_episodes=("success", "count"),
        )
        .reset_index()
        .sort_values(["radius_bin", "mode_norm"])
    )
    grp["mode_label"] = grp["mode_norm"].map(LABELS)
    return grp


# métricas de limites/velocidade/torque/suavidade (colunas de eval_rl.py):
# (coluna, rótulo do eixo, ylim)
MOTION_METRICS = [
    ("tau_sat_rate", "Passos com torque saturado (fração)", None),
    ("omega_near_rate", "Passos com |ω| ≥ 90% do limite (fração)", None),
    ("theta_near_limit_rate", "Passos perto do limite articular (fração)", None),
    ("cmd_over_limit_rate", "Passos com comando além do limite (fração)", None),
    ("jerk_rms", "Jerk RMS (rad/s³)", None),
    ("action_delta_mean", "Variação média da ação ‖Δa‖", None),
]
# colunas só da tabela (picos quase constantes, pouco informativos em gráfico)
MOTION_TABLE_ONLY = ["peak_tau", "peak_omega", "peak_abs_theta", "omega_over_rate"]


def build_motion_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Médias por seed e, sobre as seeds, média ± desvio por modo."""
    cols = [c for c, _, _ in MOTION_METRICS] + MOTION_TABLE_ONLY
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[plot_eval] Colunas {missing} ausentes (reavalie com o eval_rl.py "
              "atual); motion ignorado.")
        return None
    per_seed = df.groupby(["seed", "mode_norm"], as_index=False)[cols].mean()
    rows = []
    for mode in MODES:
        sub = per_seed[per_seed["mode_norm"] == mode]
        row = {"mode": LABELS[mode], "n_seeds": len(sub)}
        for c in cols:
            row[f"{c}_mean"], row[f"{c}_std"] = _mean_std(sub[c])
        rows.append(row)
    return pd.DataFrame(rows), per_seed


# ========= HELPERS DE PLOT =========
def save_fig(fig: plt.Figure, out_dir: str, fname: str) -> None:
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_eval] Figura salva em: {path}")


def seed_values(seed_summary: pd.DataFrame, mode: str, col: str) -> np.ndarray:
    return seed_summary[seed_summary["mode_norm"] == mode][col].values


def draw_mean_std_bars(ax, seed_summary: pd.DataFrame, metric) -> None:
    """Barras pure x pirl com média ± desvio calculados sobre as seeds."""
    col, ylabel, ylim = metric
    pure, pirl = (seed_values(seed_summary, m, col) for m in MODES)
    means = [np.nanmean(pure), np.nanmean(pirl)]
    stds = [np.nanstd(pure, ddof=1), np.nanstd(pirl, ddof=1)]
    ax.bar(np.arange(2), means, yerr=stds, tick_label=[LABELS[m] for m in MODES], capsize=5)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def available_metrics(seed_summary: pd.DataFrame, metrics: list) -> list:
    return [m for m in metrics if m[0] in seed_summary.columns]


# ========= FIGURAS: por seed =========
def fig_boxplot(seed_summary, out_dir, **_) -> None:
    fig, axes = plt.subplots(1, 2)
    for ax, (col, ylabel, _) in zip(axes, (M_SUCCESS, M_DIST)):
        ax.boxplot(
            [seed_values(seed_summary, m, col) for m in MODES],
            tick_labels=[LABELS[m] for m in MODES],
        )
        ax.set_ylabel(ylabel)
    save_fig(fig, out_dir, "fig1_boxplot_combined.pdf")


def fig_seed_grid(seed_summary, out_dir, **_) -> None:
    """Grid 2x2 com barras pure x pirl por seed (sucesso, distância, torque, energia)."""
    metrics = available_metrics(seed_summary, [M_SUCCESS, M_DIST, M_TAU, M_ENERGY])
    if len(metrics) < 4:
        print("[plot_eval] Sem torque/energia; seed-grid ignorado.")
        return
    seeds = sorted(seed_summary["seed"].unique())
    x, width = np.arange(len(seeds)), 0.4

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, (col, ylabel, ylim) in zip(axes.ravel(), metrics):
        for off, mode in ((-width / 2, "pure"), (width / 2, "pirl")):
            sub = seed_summary[seed_summary["mode_norm"] == mode].set_index("seed")[col]
            ax.bar(x + off, [sub.get(sd, np.nan) for sd in seeds], width, label=LABELS[mode])
        ax.set_xticks(x)
        ax.set_xticklabels([str(sd) for sd in seeds])
        ax.set_xlabel("Seed (modelo)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="best")
    save_fig(fig, out_dir, "fig_per_seed_grid.pdf")


# ========= FIGURAS: resumo entre seeds (média ± dp) =========
def fig_summary(seed_summary, out_dir, **_) -> None:
    metrics = available_metrics(seed_summary, [M_SUCCESS, M_DIST, M_TAU, M_ENERGY])
    n_cols = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    for ax, metric in zip(np.atleast_1d(axes).flatten(), metrics):
        draw_mean_std_bars(ax, seed_summary, metric)
    save_fig(fig, out_dir, "fig3_resumo_barras.pdf")


def fig_tradeoff(seed_summary, out_dir, **_) -> None:
    if "energy" not in seed_summary.columns:
        print("[plot_eval] Sem energia; tradeoff ignorado.")
        return
    fig, ax = plt.subplots()
    for mode in MODES:
        sub = seed_summary[seed_summary["mode_norm"] == mode]
        ax.scatter(sub["energy"], sub["success_rate"], label=LABELS[mode])
        for xv, yv, s in zip(sub["energy"], sub["success_rate"], sub["seed"]):
            ax.text(xv, yv + 0.02, str(int(s)), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Energia média por episódio")
    ax.set_ylabel("Taxa de sucesso")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.text(0.02, 0.02, "Números indicam seeds", transform=ax.transAxes,
            fontsize=8, ha="left", va="bottom")
    save_fig(fig, out_dir, "fig4_tradeoff_energia_sucesso.pdf")


def fig_target_distance(target_table, out_dir, **_) -> None:
    if target_table is None:
        return
    x, width = np.arange(len(DIST_BIN_LABELS)), 0.35
    fig, ax = plt.subplots()
    for off, mode in ((-width / 2, "pure"), (width / 2, "pirl")):
        s = target_table[target_table["mode_norm"] == mode].set_index("radius_bin")["success_rate"]
        ax.bar(x + off, [float(s.get(lbl, np.nan)) for lbl in DIST_BIN_LABELS],
               width, label=LABELS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_BIN_LABELS, rotation=45)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Taxa de sucesso")
    ax.set_xlabel("Distância inicial da ponta ao alvo (m)")
    ax.legend()
    save_fig(fig, out_dir, "fig_target_distance_success.pdf")


def fig_motion(motion, out_dir, **_) -> None:
    """Barras pure x pirl (média ± desvio entre seeds) das métricas de movimento."""
    if motion is None:
        return
    _, per_seed = motion
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, ylabel, ylim) in zip(axes.ravel(), MOTION_METRICS):
        vals = [per_seed[per_seed["mode_norm"] == m][col].values for m in MODES]
        means = [np.nanmean(v) for v in vals]
        stds = [np.nanstd(v, ddof=1) for v in vals]
        ax.bar(np.arange(2), means, yerr=stds, tick_label=[LABELS[m] for m in MODES],
               capsize=5, color=["C0", "C1"])
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
    save_fig(fig, out_dir, "fig_limites_movimento.pdf")


FIGURES = {
    "boxplot": fig_boxplot,
    "summary": fig_summary,
    "tradeoff": fig_tradeoff,
    "seed-grid": fig_seed_grid,
    "target-distance": fig_target_distance,
    "motion": fig_motion,
}


# ========= CLI =========
def save_table(table: pd.DataFrame, tables_dir: str, fname: str) -> None:
    path = os.path.join(tables_dir, fname)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[plot_eval] Tabela salva em: {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tabelas e figuras de avaliação pure x pirl a partir dos CSVs de eval_rl.py."
    )
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--last-seed", type=int, default=10)
    ap.add_argument(
        "--figures", default="all",
        help=f"'all' ou lista separada por vírgula de: {', '.join(FIGURES)}.",
    )
    ap.add_argument("--results-dir", default="results/md02",
                    help="Pasta com os CSVs de avaliação (eval_rl.py).")
    ap.add_argument("--pure-dir", default=None,
                    help="Pasta dos CSVs do pure (default: --results-dir).")
    ap.add_argument("--pirl-dir", default=None,
                    help="Pasta dos CSVs do pirl (default: --results-dir).")
    ap.add_argument("--out-dir", default="figs/figs_resultados")
    ap.add_argument("--tables-dir", default="tabelas")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.first_seed > args.last_seed:
        raise SystemExit("--first-seed deve ser <= --last-seed")

    wanted = list(FIGURES) if args.figures.strip() == "all" else [
        f.strip() for f in args.figures.split(",") if f.strip()
    ]
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        raise SystemExit(f"--figures inválido: {unknown}; use 'all' ou {list(FIGURES)}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    df, used = load_eval_data(
        args.first_seed, args.last_seed, args.results_dir, args.pure_dir, args.pirl_dir
    )
    print(f"[plot_eval] {len(used)} CSVs lidos (seeds {args.first_seed}-{args.last_seed}).")

    general = build_general_summary(df)
    seed_summary = build_seed_summary(df)
    target_table = build_target_distance_table(df)
    motion = build_motion_table(df)

    save_table(general, args.tables_dir, "resumo_geral_todos_csvs.csv")
    save_table(seed_summary, args.tables_dir, "resumo_por_seed.csv")
    if target_table is not None:
        save_table(target_table, args.tables_dir, "target_distance_success.csv")
    if motion is not None:
        save_table(motion[0], args.tables_dir, "limites_movimento.csv")

    ctx = dict(
        general=general,
        seed_summary=seed_summary,
        target_table=target_table,
        motion=motion,
        out_dir=args.out_dir,
    )
    for name in wanted:
        FIGURES[name](**ctx)


if __name__ == "__main__":
    main()
