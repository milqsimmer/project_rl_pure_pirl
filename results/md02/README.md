# results/md02 — avaliações dos treinos md02 (seeds 1 a 10)

Avaliações de `eval_rl.py` (200 episódios, `--max-steps 200`,
`--target-seed = seed`, `max_delta_theta=0.2`, `success_bonus=5.0`) dos treinos
da configuração `md02` (`lambda_a=0.01`, `success_bonus=5.0`,
`max_delta_theta=0.2`, 2M passos), seeds 1 a 10:

| Modo | run-tag | Observação |
|---|---|---|
| pure | `md02` | baseline |
| pirl | `md02_at2e4` | `alpha_tau = 2e-4`, treinado com a penalização de torque **por passo** (versão antiga do reward) |

Arquivos, para cada seed `s`:
`eval_model<s>_target<s>_mode{pure,pirl}_<s>.{csv,_summary.json,_cmd.txt}`.

São a entrada padrão de `scripts/graficos/plot_eval.py` (`--results-dir
results/md02`), que gera as tabelas de `tabelas/` e as figuras de
`figs/figs_resultados/md02/`.

Seed 0: o pure oficial está em `../baseline-oficial/` e o pirl `md02_at2e4`
em `../pirl_alpha_tau_por_passo/`. Os testes de `alpha_tau` estão em
`../pirl_alpha_tau_por_passo/` e `../pirl_alpha_tau_terminal/`.
