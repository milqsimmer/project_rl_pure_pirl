import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_summaries(
    first_seed: int,
    last_seed: int,
    out_prefix_template: str,
) -> dict:
    """Carrega os JSON de resumo gerados por eval_rl.py.

    Espera arquivos no formato:
        <out_prefix_template.format(seed=SEED)>_<SEED>_summary.json

    Cada arquivo deve conter uma lista de dicts com campos como:
        mode, seed, success_rate, mean_final_dist, mean_ep_len,
        mean_tau_effort, mean_energy, etc.
    """

    data: dict[str, dict[str, list[float]]] = {}

    seeds = list(range(first_seed, last_seed + 1))

    for seed in seeds:
        base_prefix = out_prefix_template.format(seed=seed)
        json_path = f"{base_prefix}_{seed}_summary.json"
        if not os.path.isfile(json_path):
            print(f"[plot_eval_compare] Aviso: arquivo nao encontrado: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)

        # espera lista de dicts, tipicamente um para 'pure' e outro para 'pirl'
        for s in summaries:
            mode = s.get("mode")
            if mode is None:
                continue

            if mode not in data:
                data[mode] = {
                    "seed": [],
                    "success_rate": [],
                    "mean_final_dist": [],
                    "mean_ep_len": [],
                    "mean_tau_effort": [],
                    "mean_energy": [],
                }

            data[mode]["seed"].append(seed)
            data[mode]["success_rate"].append(float(s.get("success_rate", float("nan"))))
            data[mode]["mean_final_dist"].append(
                float(s.get("mean_final_dist", float("nan")))
            )
            data[mode]["mean_ep_len"].append(float(s.get("mean_ep_len", float("nan"))))
            data[mode]["mean_tau_effort"].append(
                float(s.get("mean_tau_effort", float("nan")))
            )
            data[mode]["mean_energy"].append(
                float(s.get("mean_energy", float("nan")))
            )

    return data


def plot_bars_per_seed(
    seeds: list[int],
    pure_vals: list[float],
    pirl_vals: list[float],
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    x = np.arange(len(seeds))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, pure_vals, width, label="pure")
    ax.bar(x + width / 2, pirl_vals, width, label="pirl")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed (modelo)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_eval_compare] Figura salva em: {out_path}")


def plot_grid(
    seeds: list[int],
    pure_sr: list[float],
    pirl_sr: list[float],
    pure_ret: list[float],
    pirl_ret: list[float],
    pure_tau: list[float],
    pirl_tau: list[float],
    pure_energy: list[float],
    pirl_energy: list[float],
    out_path: str,
) -> None:
    x = np.arange(len(seeds))
    width = 0.4

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # sucesso
    ax1.bar(x - width / 2, pure_sr, width, label="pure")
    ax1.bar(x + width / 2, pirl_sr, width, label="pirl")
    ax1.set_ylabel("taxa de sucesso")
    ax1.set_title("Sucesso por seed")
    ax1.legend(loc="best")

    # retorno (aqui usamos -mean_final_dist como proxy simples)
    ax2.bar(x - width / 2, [-d for d in pure_ret], width, label="pure")
    ax2.bar(x + width / 2, [-d for d in pirl_ret], width, label="pirl")
    ax2.set_ylabel("-distancia final media")
    ax2.set_title("Retorno proxy (-dist)")
    ax2.legend(loc="best")

    # torque medio
    ax3.bar(x - width / 2, pure_tau, width, label="pure")
    ax3.bar(x + width / 2, pirl_tau, width, label="pirl")
    ax3.set_ylabel("tau_effort medio")
    ax3.set_title("Esforco medio em torque")
    ax3.legend(loc="best")

    # energia media
    ax4.bar(x - width / 2, pure_energy, width, label="pure")
    ax4.bar(x + width / 2, pirl_energy, width, label="pirl")
    ax4.set_ylabel("energia media")
    ax4.set_title("Energia media por episodio")
    ax4.legend(loc="best")

    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("seed (modelo)")

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_eval_compare] Figura salva em: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plota metricas de avaliacao (sucesso, distancia final, torque, "
            "energia) comparando pure vs pirl por seed a partir dos JSON "
            "de resumo gerados por eval_rl.py."
        )
    )
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--last-seed", type=int, default=4)
    ap.add_argument(
        "--out-prefix-template",
        type=str,
        default="results/eval_official_seed{seed}",
        help=(
            "Template do prefixo usado em eval_rl.py --out. "
            "Ex.: results/eval_official_seed{seed} ou results/eval_model{seed}_target5_modeboth."
        ),
    )
    ap.add_argument(
        "--plot-prefix",
        type=str,
        default="results/plot_eval",
        help="Prefixo para salvar as figuras de comparacao.",
    )

    args = ap.parse_args()

    data = load_summaries(args.first_seed, args.last_seed, args.out_prefix_template)

    if "pure" not in data or "pirl" not in data:
        print("[plot_eval_compare] Nao encontrou dados para pure e pirl.")
        return

    seeds_pure = data["pure"]["seed"]
    seeds_pirl = data["pirl"]["seed"]
    if seeds_pure != seeds_pirl:
        print(
            "[plot_eval_compare] Aviso: seeds de pure e pirl nao coincidem; "
            "vou intersectar as listas."
        )
    seeds = sorted(set(seeds_pure) & set(seeds_pirl))
    if not seeds:
        print("[plot_eval_compare] Nenhuma seed em comum entre pure e pirl.")
        return

    def vals(mode: str, key: str) -> list[float]:
        res = []
        for s in seeds:
            for i, seed in enumerate(data[mode]["seed"]):
                if seed == s:
                    res.append(data[mode][key][i])
                    break
        return res

    pure_sr = vals("pure", "success_rate")
    pirl_sr = vals("pirl", "success_rate")
    pure_dist = vals("pure", "mean_final_dist")
    pirl_dist = vals("pirl", "mean_final_dist")
    pure_tau = vals("pure", "mean_tau_effort")
    pirl_tau = vals("pirl", "mean_tau_effort")
    pure_energy = vals("pure", "mean_energy")
    pirl_energy = vals("pirl", "mean_energy")

    # 1) barra: taxa de sucesso por seed
    plot_bars_per_seed(
        seeds,
        pure_sr,
        pirl_sr,
        ylabel="taxa de sucesso",
        title="Taxa de sucesso por seed (pure vs pirl)",
        out_path=f"{args.plot_prefix}_success_rate.png",
    )

    # 2) barra: distancia final media por seed
    plot_bars_per_seed(
        seeds,
        pure_dist,
        pirl_dist,
        ylabel="distancia final media",
        title="Distancia final media por seed (pure vs pirl)",
        out_path=f"{args.plot_prefix}_final_dist.png",
    )

    # 3) barra: torque medio (tau_effort) por seed
    plot_bars_per_seed(
        seeds,
        pure_tau,
        pirl_tau,
        ylabel="tau_effort medio",
        title="Esforco medio em torque por seed (pure vs pirl)",
        out_path=f"{args.plot_prefix}_tau_effort.png",
    )

    # 4) barra: energia media por seed
    plot_bars_per_seed(
        seeds,
        pure_energy,
        pirl_energy,
        ylabel="energia media por episodio",
        title="Energia media por seed (pure vs pirl)",
        out_path=f"{args.plot_prefix}_energy.png",
    )

    # 5) grid com as 4 metricas principais
    plot_grid(
        seeds,
        pure_sr,
        pirl_sr,
        pure_dist,
        pirl_dist,
        pure_tau,
        pirl_tau,
        pure_energy,
        pirl_energy,
        out_path=f"{args.plot_prefix}_grid.png",
    )


if __name__ == "__main__":
    main()
