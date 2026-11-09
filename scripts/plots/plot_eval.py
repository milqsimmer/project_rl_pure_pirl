"""Gráficos de avaliação PPO baseados nos CSVs gerados por eval_rl.py.

Uso (na raiz do projeto):
    python scripts/plots/plot_eval.py --run-tag pilot
    python scripts/plots/plot_eval.py --pattern 'results/eval_*_pilot.csv'

As médias de métricas são calculadas por seed; não interpretar episódios
como seeds independentes. Requer mesmo target_seed para comparações pareadas.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("success", "Taxa de sucesso", "sucesso", "proporção"),
    ("safe_success", "Taxa de sucesso seguro", "sucesso_seguro", "proporção"),
    ("final_distance", "Distância final", "distancia_final", "m"),
    ("elapsed_s", "Tempo por episódio", "tempo", "s"),
    ("torque_abs_integral", "Esforço articular integrado", "torque_integrado", "N·m·s"),
    (
        "torque_sq_integral",
        "Torque quadrático integrado",
        "torque_quadratico",
        "(N·m)²·s",
    ),
    ("energy_abs", "Energia mecânica absoluta", "energia", "J"),
    ("peak_tau", "Pico de torque", "pico_torque", "N·m"),
    ("peak_omega", "Pico de velocidade", "pico_velocidade", "rad/s"),
    (
        "torque_envelope_rate",
        "Fração de passos físicos fora do envelope de torque",
        "envelope_torque",
        "fração",
    ),
    (
        "velocity_envelope_rate",
        "Fração de passos físicos fora do envelope de velocidade",
        "envelope_velocidade",
        "fração",
    ),
    (
        "position_envelope_rate",
        "Fração de passos físicos fora do envelope articular",
        "envelope_articular",
        "fração",
    ),
    (
        "tau_sat_rate",
        "Fração de passos físicos com saturação de torque",
        "saturacao_torque",
        "fração",
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default="pilot")
    ap.add_argument("--pattern", default=None, help="Glob dos CSVs de avaliação")
    ap.add_argument("--out-dir", default="figs/eval")
    ap.add_argument("--distance-bins", type=int, default=5)
    args = ap.parse_args()
    if args.distance_bins < 2:
        ap.error("--distance-bins precisa ser >= 2")
    pattern = args.pattern or f"results/eval_*_{args.run_tag}.csv"
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        ap.error(f"Nenhum arquivo de avaliação encontrado: {pattern}")
    chunks = []
    for path in paths:
        try:
            d = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError) as exc:
            print(f"[AVISO] Ignorando {path}: {exc}")
            continue
        if d.empty or not {"seed", "episode", "success", "final_distance"}.issubset(
            d.columns
        ):
            print(f"[AVISO] Sem colunas mínimas: {path}")
            continue
        d["source_file"] = path.name
        chunks.append(d)
        print(f"{path}: {len(d)} episódios")
    if not chunks:
        ap.error("Não há CSVs de avaliação válidos")
    df = pd.concat(chunks, ignore_index=True)
    if "target_seed" in df.columns and df["target_seed"].nunique() > 1:
        print(
            "[ATENÇÃO] target_seed difere entre arquivos: não interpretar diferenças episódio a episódio como pareadas."
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df["seed"] = df["seed"].astype(str)
    # Métricas por seed, nunca IC calculado tratando episódios como runs independentes.
    records = []
    for name, title, slug, unit in METRICS:
        if name not in df:
            continue
        d = df[["seed", name]].copy()
        d[name] = pd.to_numeric(d[name], errors="coerce")
        grouped = d.groupby("seed")[name].agg(["mean", "count"]).sort_index()
        if grouped["count"].sum() == 0:
            continue
        for seed, row in grouped.iterrows():
            records.append(
                {
                    "seed": seed,
                    "metric": name,
                    "mean": row["mean"],
                    "episodes": int(row["count"]),
                }
            )
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(grouped))
        ax.bar(x, grouped["mean"], alpha=0.65)
        for i, value in enumerate(grouped["mean"]):
            if np.isfinite(value):
                ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels=[f"Seed {s}" for s in grouped.index])
        ax.set(title=title + " — média por seed", ylabel=unit)
        if name in (
            "success",
            "safe_success",
            "torque_envelope_rate",
            "velocity_envelope_rate",
            "position_envelope_rate",
            "tau_sat_rate",
        ):
            ax.set_ylim(0, 1.08)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"eval_{slug}.png", dpi=165)
        plt.close(fig)
    if records:
        pd.DataFrame.from_records(records).to_csv(
            out / "resumo_por_seed.csv", index=False
        )

    # Comparação tarefa x esforço por episódio, distinguindo sucesso e falha.
    if {"torque_abs_integral", "final_distance", "success"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for success, label in [(0, "Falha"), (1, "Sucesso")]:
            pts = df[df["success"] == success]
            ax.scatter(
                pts["torque_abs_integral"],
                pts["final_distance"],
                s=20,
                alpha=0.55,
                label=label,
            )
        ax.set(
            xlabel="Esforço articular integrado (N·m·s)",
            ylabel="Distância final (m)",
            title="Compromisso entre esforço e precisão — episódios",
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "eval_esforco_vs_distancia.png", dpi=165)
        plt.close(fig)

    # Facilidade dos alvos: distância inicial até a ponta, mesma discretização em todas as seeds.
    if {"tip_init_dist", "success"}.issubset(df.columns):
        dist = pd.to_numeric(df["tip_init_dist"], errors="coerce")
        valid = df.loc[dist.notna()].copy()
        valid["tip_init_dist"] = dist[dist.notna()]
        if (
            len(valid) >= args.distance_bins
            and valid["tip_init_dist"].max() > valid["tip_init_dist"].min()
        ):
            edges = np.linspace(
                valid["tip_init_dist"].min() - 1e-9,
                valid["tip_init_dist"].max() + 1e-9,
                args.distance_bins + 1,
            )
            valid["bin"] = pd.cut(
                valid["tip_init_dist"], bins=edges, include_lowest=True
            )
            rates = (
                valid.groupby(["seed", "bin"], observed=False)["success"]
                .agg(["mean", "count"])
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            centers = (edges[:-1] + edges[1:]) / 2
            for seed, sub in rates.groupby("seed"):
                sub = sub.sort_values("bin")
                y = sub["mean"].to_numpy(dtype=float, copy=True)
                y[sub["count"].to_numpy() == 0] = np.nan
                ax.plot(centers, y, marker="o", label=f"Seed {seed}")
            ax.set(
                xlabel="Distância inicial da ponta ao alvo (m)",
                ylabel="Taxa de sucesso",
                title="Sucesso por distância inicial do alvo",
                ylim=(-0.02, 1.02),
            )
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "eval_sucesso_por_distancia_inicial.png", dpi=165)
            plt.close(fig)
            print(
                "[NOTA] Verifique contagens por faixa antes de interpretar curvas com poucos episódios."
            )
    print(f"Gráficos de avaliação salvos em {out.resolve()}")


if __name__ == "__main__":
    main()
