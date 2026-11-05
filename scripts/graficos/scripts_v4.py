import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "tabelas/resumo_geral_todos_csvs.csv"
OUT_PATH = "figs/figs_resultados/fig_sucesso_medio_pure_pirl.pdf"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df = pd.read_csv(CSV_PATH)
# Normaliza modo
df["mode_norm"] = (
    df["mode"].str.lower().map(lambda s: "pirl" if "pirl" in s else "pure")
)
df_plot = df[df["mode_norm"].isin(["pure", "pirl"])]
modes = ["pure", "pirl"]
labels = ["RL puro", "PIRL (PI-reward)"]
success = [float(df_plot[df_plot["mode_norm"] == m]["success_rate_mean"].iloc[0]) for m in modes]
fig, ax = plt.subplots(figsize=(4, 4))
x = [0, 1]
ax.bar(x, success, color=["C0", "C1"])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Taxa de sucesso média")
ax.set_ylim(0.0, 1.0)
ax.set_title("Taxa de sucesso média (seeds 1–10)")
for xi, s in zip(x, success):
    ax.text(xi, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Figura salva em: {OUT_PATH}")


CSV_PATH = "tabelas/resumo_geral_todos_csvs.csv"
OUT_PATH = "figs/figs_resultados/fig_torque_energia_medio_pure_pirl.pdf"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df = pd.read_csv(CSV_PATH)
df["mode_norm"] = (
    df["mode"].str.lower().map(lambda s: "pirl" if "pirl" in s else "pure")
)
df_plot = df[df["mode_norm"].isin(["pure", "pirl"])]
modes = ["pure", "pirl"]
labels = ["RL puro", "PIRL (PI-reward)"]
tau = [float(df_plot[df_plot["mode_norm"] == m]["tau_effort_mean"].iloc[0]) for m in modes]
energy = [float(df_plot[df_plot["mode_norm"] == m]["energy_mean"].iloc[0]) for m in modes]
x = [0, 1]
width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# Torque
ax1.bar(x, tau, color=["C0", "C1"])
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=0)
ax1.set_ylabel("Esforço médio em torque")
ax1.set_title("Torque médio por episódio")
# Energia
ax2.bar(x, energy, color=["C0", "C1"])
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=0)
ax2.set_ylabel("Energia média")
ax2.set_title("Energia média por episódio")
fig.tight_layout()
fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Figura salva em: {OUT_PATH}")


CSV_PATH = "tabelas/safety_metrics.csv"
OUT_PATH = "figs/figs_resultados/fig_sucesso_vs_sucesso_seguro_pure_pirl.pdf"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df = pd.read_csv(CSV_PATH)
df["mode_norm"] = (
    df["mode"].str.lower().map(lambda s: "pirl" if "pirl" in s else "pure")
)
df_plot = df[df["mode_norm"].isin(["pure", "pirl"])]
modes = ["pure", "pirl"]
labels = ["RL puro", "PIRL (PI-reward)"]
succ = [float(df_plot[df_plot["mode_norm"] == m]["success_rate"].iloc[0]) for m in modes]
safe_succ = [
    float(df_plot[df_plot["mode_norm"] == m]["safe_success_rate"].iloc[0]) for m in modes
]
x = [0, 1]
width = 0.35
fig, ax = plt.subplots(figsize=(5, 4))
# barra de sucesso total (mais clara)
ax.bar(
    [xi - width / 2 for xi in x],
    succ,
    width,
    label="Sucesso total",
    color="C0",
    alpha=0.4,
)
# barra de sucesso seguro (mais escura, mesma posição de modo)
ax.bar(
    [xi + width / 2 for xi in x], safe_succ, width, label="Sucesso seguro", color="C0"
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Taxa de sucesso")
ax.set_ylim(0.0, 1.0)
ax.set_title("Sucesso total vs sucesso seguro (seeds 1–10)")
ax.legend(loc="best")
for xi, s, ss in zip(x, succ, safe_succ):
    ax.text(xi - width / 2, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(
        xi + width / 2, ss + 0.01, f"{ss:.2f}", ha="center", va="bottom", fontsize=8
    )
fig.tight_layout()
fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Figura salva em: {OUT_PATH}")
