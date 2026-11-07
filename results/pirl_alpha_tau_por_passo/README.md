# results/pirl_alpha_tau_por_passo — varredura de alpha_tau (torque por passo)

Piloto de escolha do `alpha_tau` do PIRL na **versão antiga do reward**, em que
a penalização de torque era aplicada **a cada passo**
(`reward -= alpha_tau * tau_sum`). Foi o piloto que levou ao `alpha_tau = 2e-4`
das runs oficiais (`md02_at2e4`); ver `docs/plano-24-10.md`.

Só seed 0. Config: `lambda_a=0.01`, `success_bonus=5.0`, `max_delta_theta=0.2`,
2M passos, run-tag `md02_at<valor>`. Modelos em
`runs/runs_pirl/seed_0_md02_at{2e4,3e4,5e4}/`.

## Arquivos

`eval_seed0_md02_at<valor>_0.{csv,_summary.json,_cmd.txt}`, 200 episódios cada.

## Resultados (seed 0)

| alpha_tau | Sucesso | Dist. final | Torque médio | Energia média |
|---|---|---|---|---|
| (pure md02, referência) | 0.980 | 0.061 | 2.02 | 1.53 |
| `2e-4` (`at2e4`) | 0.835 | 0.111 | 1.51 | 1.45 |
| `3e-4` (`at3e4`) | 0.745 | 0.133 | 2.12 | 2.13 |
| `5e-4` (`at5e4`) | 0.655 | 0.181 | 1.51 | 1.67 |

`2e-4` foi o melhor compromisso: mantém sucesso razoável e reduz o torque em
cerca de 25% em relação ao pure. Com `3e-4` e `5e-4` o sucesso cai e o esforço
não melhora de forma consistente.

O `eval_seed0_md02_at2e4_0` é também a avaliação do pirl oficial da seed 0; as
seeds 1 a 10 do pirl oficial estão em `results/md02/eval_model<seed>_target<seed>_modepirl_<seed>.*`.

## Arquivo sem resultado

- `eval_model1_target1_modeboth_1_cmd.txt`: só o comando de uma avaliação
  `--mode both` (seed 1, run-tag `md02_at2e4`), sem CSV nem summary.
  Provavelmente falhou por não existir um modelo pure com esse tag.
