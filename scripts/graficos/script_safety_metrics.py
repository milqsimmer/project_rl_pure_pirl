"""Calcula métricas de "segurança" a partir dos CSVs de eval.

Lê arquivos no formato mais novo gerado por eval_rl.py:

    results/eval_model<seed>_target<seed>_mode{pure,pirl}_<seed>.csv

e calcula, para cada modo (pure, pirl):

- success_rate               = média de success (0/1)
- safe_success_rate          = fração de episódios com sucesso E torque/energia
                               abaixo de limites "seguros" e duração <= limiar
- P_tau_excesso              = fração de episódios com mean_tau_sum > tau_safe
- P_energy_excesso           = fração de episódios com episode_energy > E_safe

Os limites "seguros" são definidos a partir dos episódios do modo pure
no intervalo de seeds escolhido:

- tau_safe = percentil TAU_PCT do mean_tau_sum do pure
- E_safe   = percentil E_PCT   do episode_energy do pure

Uso típico (seeds 1 a 10):

    python scripts/graficos/script_safety_metrics.py --first-seed 1 --last-seed 10

Gera uma tabela CSV em tabelas/safety_metrics.csv com uma linha por modo.
"""

import argparse
import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd


OUT_TABLE = "tabelas/safety_metrics.csv"


def load_eval_csvs(first_seed: int, last_seed: int) -> pd.DataFrame:
    """Carrega todos os CSVs de eval_model*_target*_mode*_* para o range de seeds.

    Retorna um DataFrame único com colunas:
        mode, seed, success, length, mean_tau_sum, episode_energy, ...
    """

    pattern = "results/eval_model*_target*_mode*_*.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"Nenhum CSV encontrado com padrão: {pattern}")

    records = []
    wanted = set(range(first_seed, last_seed + 1))

    for path in paths:
        name = os.path.basename(path)

        # Extrai seed e modo do nome do arquivo
        # ex.: eval_model1_target1_modepure_1.csv
        try:
            base, _ = os.path.splitext(name)
            parts = base.split("_")
            # formato esperado: ["eval", "model<seed>", "target<seed>", "mode<mode>", "<seed>"]
            seed_str = parts[1].replace("model", "")
            seed = int(seed_str)
            if seed not in wanted:
                continue
        except Exception:
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        df = df.copy()
        df["seed"] = seed
        records.append(df)

    if not records:
        raise SystemExit(
            f"Nenhum CSV de eval_model*_target*_mode*_* para seeds {first_seed}..{last_seed}"
        )

    all_df = pd.concat(records, ignore_index=True)

    # normaliza colunas principais
    required_cols = [
        "mode",
        "seed",
        "success",
        "length",
        "mean_tau_sum",
        "energy",
    ]
    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise SystemExit(
            f"Colunas faltando nos CSVs de eval: {missing}. "
            f"Precisamos de {required_cols} para métricas de segurança."
        )

    # garante tipos numéricos
    num_cols = ["success", "length", "mean_tau_sum", "energy"]
    all_df[num_cols] = all_df[num_cols].apply(pd.to_numeric, errors="coerce")
    all_df = all_df.dropna(subset=["mode", "success", "length", "mean_tau_sum", "energy"])

    return all_df


def compute_safety_thresholds(df_pure: pd.DataFrame, tau_pct: float, e_pct: float) -> Tuple[float, float]:
    """Calcula tau_safe e E_safe a partir do modo pure.

    tau_safe = percentil tau_pct de mean_tau_sum
    E_safe   = percentil e_pct   de episode_energy
    """

    tau_safe = float(np.percentile(df_pure["mean_tau_sum"].values, tau_pct))
    e_safe = float(np.percentile(df_pure["energy"].values, e_pct))
    return tau_safe, e_safe


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Calcula métricas de segurança (safe_success_rate e excedência "
            "de torque/energia) para pure vs pirl a partir dos CSVs "
            "eval_model*_target*_mode*_* gerados pelo eval_rl.py."
        )
    )
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--last-seed", type=int, default=10)
    ap.add_argument(
        "--tau-percentile",
        type=float,
        default=75.0,
        help=(
            "Percentil usado para definir tau_safe a partir do modo pure. "
            "Default=75.0 (3o quartil)."
        ),
    )
    ap.add_argument(
        "--energy-percentile",
        type=float,
        default=75.0,
        help=(
            "Percentil usado para definir E_safe a partir do modo pure. "
            "Default=75.0 (3o quartil)."
        ),
    )
    ap.add_argument(
        "--max-safe-length",
        type=int,
        default=180,
        help=(
            "Comprimento maximo considerado seguro (em passos). "
            "Episodios com length > este valor nao contam como safe_success."
        ),
    )

    args = ap.parse_args()

    if args.first_seed > args.last_seed:
        raise SystemExit("first-seed deve ser <= last-seed")

    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    df = load_eval_csvs(args.first_seed, args.last_seed)

    df["mode_norm"] = df["mode"].str.lower().map(
        lambda s: "pirl" if "pirl" in s else "pure"
    )

    df_pure = df[df["mode_norm"] == "pure"]
    df_pirl = df[df["mode_norm"] == "pirl"]

    if df_pure.empty or df_pirl.empty:
        raise SystemExit("Precisamos de dados tanto para pure quanto para pirl.")

    tau_safe, e_safe = compute_safety_thresholds(
        df_pure, args.tau_percentile, args.energy_percentile
    )

    print(
        f"[safety_metrics] tau_safe (P{args.tau_percentile} pure) = {tau_safe:.4f}, "
        f"E_safe (P{args.energy_percentile} pure) = {e_safe:.4f}"
    )

    rows = []
    for mode in ["pure", "pirl"]:
        sub = df[df["mode_norm"] == mode]
        if sub.empty:
            continue

        succ_rate = float(sub["success"].mean())

        safe_mask = (
            (sub["success"] == 1)
            & (sub["mean_tau_sum"] <= tau_safe)
            & (sub["energy"] <= e_safe)
            & (sub["length"] <= args.max_safe_length)
        )
        safe_success_rate = float(safe_mask.mean())

        p_tau_excess = float((sub["mean_tau_sum"] > tau_safe).mean())
        p_e_excess = float((sub["energy"] > e_safe).mean())

        rows.append(
            {
                "mode": mode,
                "success_rate": succ_rate,
                "safe_success_rate": safe_success_rate,
                "tau_safe": tau_safe,
                "E_safe": e_safe,
                "P_tau_excesso": p_tau_excess,
                "P_energy_excesso": p_e_excess,
                "first_seed": args.first_seed,
                "last_seed": args.last_seed,
                "max_safe_length": args.max_safe_length,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_TABLE, index=False)
    print(f"[safety_metrics] Tabela salva em: {OUT_TABLE}")


if __name__ == "__main__":
    main()
