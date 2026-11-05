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

**Valor escolhido: `alpha_tau = 0.3`** (run-tag `md02_at03_term`). Ele foi
avaliado nas avaliações `*_term` (`alpha_tau` = 0.1, 0.3 e 0.5), que continuam
esta varredura com valores maiores e ainda estão soltas na raiz de `results/`
porque pertencem a experimentos em andamento (a validação do 0.3 em mais seeds
ainda roda); devem ser movidas para esta pasta quando terminarem.

## Arquivos

`eval_pirl_seed<s>_md02_at<valor>_<s>.{csv,_summary.json,_cmd.txt}`, 200
episódios cada.

## Resultados (taxa de sucesso)

| seed | at1e2 (0.01) | at2e2 (0.02) | at5e2 (0.05) | pure md02 (referência) |
|---|---|---|---|---|
| 0 | 0.935 | 0.770 | 0.765 | 0.980 |
| 1 | 0.920 | 0.930 | 0.805 | 0.725 |
| 2 | 0.945 | – | – | 0.875 |
| 3 | 0.695 | – | – | 0.970 |
| 4 | 0.765 | – | – | 0.720 |

Com `alpha_tau = 0.01`, nas seeds 0 a 4: sucesso médio 0.852 (pure: 0.854),
distância final 0.094 (pure: 0.099), torque médio 1.50 (pure: 1.63, cerca de 8%
menos) e energia média 1.45 (pure: 1.48). Ou seja, sucesso equivalente ao pure
com pouco menos esforço. Valores maiores (0.02 e 0.05) foram testados só nas
seeds 0 e 1 e deram sucesso menor na seed 0.

## Arquivos sem resultado

- `eval_seed0_md02_at2e2_0_cmd.txt` e `eval_seed1_md02_at2e2_1_cmd.txt`: só o
  comando de avaliações `--mode both`, sem CSV nem summary. Provavelmente
  falharam por não existir um modelo pure com esse tag.
