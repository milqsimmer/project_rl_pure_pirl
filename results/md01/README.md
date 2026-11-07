# results/md01 — primeira rodada de avaliação (pré-md02)

Avaliações de `eval_rl.py` dos modelos PPO das seeds 0 a 4, treinados **antes**
do ajuste de hiperparâmetros que originou a configuração `md02`
(`max_delta_theta=0.2`, `success_bonus=5.0`, `lambda_a=0.01`). Fica como
registro histórico dessa fase; nenhum script atual lê esta pasta.

## O que foi avaliado

- Modos `pure` e `pirl` na mesma execução (`--mode both`), 200 episódios por
  modo, `--max-steps 200`.
- Modelos com run-tag `at2e4` (`alpha_tau = 2e-4` no pirl; o pure usa o mesmo
  tag só por convenção de nome de pasta).
- `success_bonus = 0.0` na avaliação, como no treino desta fase.
- O `eval_rl.py` ainda não tinha `--max-delta-theta`, então valia o padrão do
  ambiente na época (0.1).

## Arquivos

Para cada par `(model_seed, target_seed)`:

- `eval_model<model_seed>_target<target_seed>_modeboth_<model_seed>.csv`:
  métricas por episódio.
- `..._summary.json`: resumo agregado por modo.
- `..._cmd.txt`: comando e argumentos usados.
- `target_seed-0-4.rar`: cópia compactada do mesmo material.

Os pares avaliados são `(s, s)` e `(s, s+5)` para `s = 0..4`, ou seja, cada
modelo foi testado na própria sequência de alvos (`target_seed = seed`) e em
uma sequência diferente (`target_seed = seed + 5`), para checar generalização
a alvos novos.

## Resultados (taxa de sucesso)

| model_seed | target_seed | pure | pirl |
|---|---|---|---|
| 0 | 0 | 0.385 | 0.405 |
| 0 | 5 | 0.430 | 0.425 |
| 1 | 1 | 0.380 | 0.525 |
| 1 | 6 | 0.335 | 0.545 |
| 2 | 2 | 0.290 | 0.290 |
| 2 | 7 | 0.375 | 0.290 |
| 3 | 3 | 0.420 | 0.380 |
| 3 | 8 | 0.420 | 0.415 |
| 4 | 4 | 0.640 | 0.440 |
| 4 | 9 | 0.640 | 0.390 |

Leitura: o sucesso é baixo (0.29 a 0.64) e o pirl não é consistentemente
melhor nem pior que o pure. O pirl usa menos torque e energia (por exemplo,
seed 0: torque médio 0.47 contra 0.55; energia 0.38 contra 0.52), mas sem ganho
claro de sucesso. Trocar o `target_seed` quase não muda o resultado. Esse
desempenho baixo motivou o ajuste de `max_delta_theta`, `success_bonus` e
`lambda_a` (ver os testes de hiperparâmetros do pure em `results/pure_hiperparametros/`),
que levou à configuração `md02`.

## Como se relaciona com o resto

- Resultados atuais (md02, seeds 1 a 10): `results/md02/eval_model<seed>_target<seed>_mode{pure,pirl}_<seed>.csv`,
  lidos por `scripts/graficos/plot_eval.py`.
- Os modelos desta fase estavam em `runs_pure/seed_<s>_at2e4/` e
  `runs_pirl/seed_<s>_at2e4/` (com `monitor.csv`, `ppo_model_<s>.zip` e
  `train_cmd.txt`). Foram removidos da árvore de trabalho na limpeza
  (commit `b8e3c50`); continuam recuperáveis no histórico do git, por exemplo
  a partir do commit `2efaa9b`.
