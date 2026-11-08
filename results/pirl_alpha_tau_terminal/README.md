# results/pirl_alpha_tau_terminal — varredura de alpha_tau (torque terminal)

Piloto de escolha do `alpha_tau` do PIRL com a penalização de torque
**terminal**: o reward por passo é igual ao do pure e, no passo em que o alvo é
atingido, desconta `alpha_tau * torque_médio_do_episódio`. É a versão do reward
da branch `v4-nova-abordagem` (commit `d182e74`, de onde vieram as runs desta
pasta). Complementa `../pirl_alpha_tau_por_passo/`, que fez o mesmo piloto com
a penalização por passo.

Config: `lambda_a=0.01`, `success_bonus=5.0`, `max_delta_theta=0.2`, 2M passos,
run-tag `md02_at<valor>`; `alpha_tau` = 0.01 (`at1e2`), 0.02 (`at2e2`) e
0.05 (`at5e2`). Modelos em `runs/runs_pirl/seed_<s>_md02_at{1e2,2e2,5e2}/`.

**Valor escolhido: `alpha_tau = 0.3`** (run-tag `md02_at03_term`). A varredura
continua com valores maiores na seed 0: `alpha_tau` = 0.1 (`at01_term`), 0.3
(`at03_term`) e 0.5 (`at05_term`), modelos em
`runs/runs_pirl/seed_0_md02_at0{1,3,5}_term/`. A validação do 0.3 nas seeds 1 a 10
está em `../md02_at03_term/` (tabelas em `../../tabelas/md02_at03_term/` e figuras em
`../../figs/figs_resultados/md02_at03_term/`), e o teste com
`alpha_tau` 1.0 e 3.0 (seed 1) em `../md02_at_sweep_seed1/`.

## Arquivos

- `eval_pirl_seed<s>_md02_at<valor>_<s>.{csv,_summary.json,_cmd.txt}` (`at1e2`,
  `at2e2`, `at5e2`), 200 episódios cada.
- `eval_seed0_md02_at0{1,3,5}_term_0.{csv,_summary.json,_cmd.txt}` (seed 0,
  `alpha_tau` 0.1, 0.3 e 0.5).

## Resultados (taxa de sucesso)

| seed | at1e2 (0.01) | at2e2 (0.02) | at5e2 (0.05) | pure md02 (referência) |
|---|---|---|---|---|
| 0 | 0.935 | 0.770 | 0.765 | 0.980 |
| 1 | 0.920 | 0.930 | 0.805 | 0.725 |
| 2 | 0.945 | – | – | 0.875 |
| 3 | 0.695 | – | – | 0.970 |
| 4 | 0.765 | – | – | 0.720 |

Seed 0 com valores maiores de `alpha_tau`: sucesso 0.795 (0.1), 0.965 (0.3) e
0.980 (0.5), contra 0.980 do pure.

Com `alpha_tau = 0.01`, nas seeds 0 a 4: sucesso médio 0.852 (pure: 0.854),
distância final 0.094 (pure: 0.099), torque médio 1.50 (pure: 1.63, cerca de 8%
menos) e energia média 1.45 (pure: 1.48). Ou seja, sucesso equivalente ao pure
com pouco menos esforço. Valores maiores (0.02 e 0.05) foram testados só nas
seeds 0 e 1 e deram sucesso menor na seed 0.

## Arquivos sem resultado

- `eval_seed0_md02_at2e2_0_cmd.txt` e `eval_seed1_md02_at2e2_1_cmd.txt`: só o
  comando de avaliações `--mode both`, sem CSV nem summary. Provavelmente
  falharam por não existir um modelo pure com esse tag.
