"""Análise de sensibilidade ao envelope de operação (70/80/90/100% dos limites).

Para cada grupo de modelos (um rótulo + um padrão de arquivos de avaliação), lê os
CSVs por episódio de eval_rl.py e calcula, por seed e depois entre seeds, a fração de
passos acima de p% do limite de velocidade, torque e ângulo articular, junto com a
taxa de sucesso. Gera:

  <tables-dir>/<name>.csv             tabela larga (média e desvio entre seeds)
  <out-dir>/fig_<name>.pdf            fração acima do envelope x % do limite, por grupo

Requer CSVs gerados pelo eval_rl.py atual (colunas omega_gt*, tau_gt*, theta_gt*).
Os limites de referência são os do URDF: 5 rad/s, 20 N.m e 3,14 rad.

Uso (a partir da raiz do projeto), um --group por modelo/configuração:

  python scripts/graficos/plot_sensibilidade.py \
      --group "pure normalizado=results/md02_nrm_pilot/eval_pure_nrm_*.csv" \
      --group "pirl x0.5=results/md02_nrm_pilot/eval_pirl_nrm_lim_x05_*.csv" \
      --group "pirl x1=results/md02_nrm_pilot/eval_pirl_nrm_lim_x1_*.csv"
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PCTS = (70, 80, 90, 100)
METRICS = [
    ("omega", "Velocidade articular (> p% de 5 rad/s)"),
    ("tau", "Torque (> p% de 20 N·m)"),
    ("theta", "Ângulo articular (> p% de 3,14 rad)"),
]
COLS = ["success"] + [f"{m}_gt{p}" for m, _ in METRICS for p in PCTS]


def seed_from_name(path: str) -> int:
    m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
    if not m:
        raise SystemExit(f"Não achei a seed no nome do arquivo: {path}")
    return int(m.group(1))


def load_group(label: str, pattern: str) -> pd.DataFrame:
    """Média por seed das colunas de interesse (uma linha por seed)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Nenhum arquivo para o grupo '{label}': {pattern}")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{f} não tem as colunas {missing}; reavalie com o eval_rl.py atual."
            )
        row = df[COLS].mean()
        row["seed"] = seed_from_name(f)
        rows.append(row)
    return pd.DataFrame(rows).set_index("seed")


def build_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for label, per_seed in groups.items():
        row = {"group": label, "n_seeds": len(per_seed)}
        for c in COLS:
            row[f"{c}_mean"] = per_seed[c].mean()
            row[f"{c}_std"] = per_seed[c].std(ddof=1) if len(per_seed) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def plot(groups: dict[str, pd.DataFrame], path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for label, per_seed in groups.items():
            means = [per_seed[f"{metric}_gt{p}"].mean() for p in PCTS]
            stds = [
                per_seed[f"{metric}_gt{p}"].std(ddof=1) if len(per_seed) > 1 else 0.0
                for p in PCTS
            ]
            ax.errorbar(PCTS, means, yerr=stds, marker="o", capsize=3,
                        label=f"{label} (sucesso {per_seed['success'].mean():.2f})")
        ax.set_xticks(PCTS)
        ax.set_xlabel("Envelope (% do limite)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fração de passos acima do envelope")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[sensibilidade] Figura salva em: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensibilidade ao envelope (70/80/90/100%).")
    ap.add_argument("--group", action="append", required=True,
                    metavar="ROTULO=GLOB",
                    help="Grupo de avaliações: rótulo e padrão de arquivos (repita o argumento).")
    ap.add_argument("--name", default="sensibilidade_envelope",
                    help="Nome-base dos arquivos de saída.")
    ap.add_argument("--out-dir", default="figs/figs_resultados")
    ap.add_argument("--tables-dir", default="tabelas")
    args = ap.parse_args()

    groups: dict[str, pd.DataFrame] = {}
    for spec in args.group:
        if "=" not in spec:
            raise SystemExit(f"--group deve ser ROTULO=GLOB, recebi: {spec}")
        label, pattern = spec.split("=", 1)
        groups[label.strip()] = load_group(label.strip(), pattern.strip())

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    table = build_table(groups)
    tpath = os.path.join(args.tables_dir, f"{args.name}.csv")
    table.to_csv(tpath, index=False, encoding="utf-8-sig")
    print(f"[sensibilidade] Tabela salva em: {tpath}")

    # resumo legível no terminal
    show = ["group", "n_seeds", "success_mean"] + [
        f"{m}_gt{p}_mean" for m in ("omega", "tau", "theta") for p in (80,)
    ]
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot(groups, os.path.join(args.out_dir, f"fig_{args.name}.pdf"))


if __name__ == "__main__":
    main()
