# results/pure_hiperparametros — avaliações dos testes de hiperparâmetros

Avaliações de `eval_rl.py` feitas durante a calibração do baseline `pure` na
seed 0 e dos checks de sanidade do `pirl`. O objetivo era escolher a
configuração que virou o baseline `md02` (ver `docs/plano-24-10.md` e a
seção "Current tuning plan" do `AGENTS.md`):

    steps = 2_000_000, learning-rate = 3e-4, lambda_a = 0.01,
    success_bonus = 5.0, max-delta-theta = 0.2, max-steps = 200

Todas as avaliações usam 200 episódios e `--max-steps 200`. Os modelos
avaliados estão em `runs/runs_pure/testes parametros/` e
`runs/runs_pirl/teste tau 0/` (o `train_cmd.txt` de cada run tem o comando de
treino), exceto onde indicado.

## 1. Varredura de hiperparâmetros do pure (seed 0)

Arquivos `eval_pure_seed0_<tag>_0.{csv,_summary.json,_cmd.txt}`. Todos com
`lambda_a=0.01` e `success_bonus=5.0` no treino, exceto o que cada linha varia:

| Tag | O que muda no treino | Sucesso | Dist. final média |
|---|---|---|---|
| `base` | padrão (`max_delta_theta=0.1`, 2M passos) | 0.620 | 0.102 |
| `la5e3` | `lambda_a = 0.005` | 0.445 | 0.148 |
| `sb8` | `success_bonus = 8.0` | 0.420 | 0.132 |
| `md015` | `max_delta_theta = 0.15` | 0.860 | 0.128 |
| `md02` | `max_delta_theta = 0.2` (**config adotada**) | **0.980** | 0.061 |
| `md02_s5e5` | md02 com 500 mil passos | 0.095 | 0.520 |
| `md02_s1e6` | md02 com 1 milhão de passos | 0.490 | 0.257 |
| `md02_s15e5` | md02 com 1,5 milhão de passos | 0.645 | 0.228 |

Leitura: aumentar `max_delta_theta` foi o ajuste que mais ajudou; reduzir
`lambda_a` ou aumentar `success_bonus` isoladamente piorou em relação ao
`base`; e o `md02` ainda precisa dos 2M passos (com menos passos o sucesso cai
muito).

## 2. Reavaliações e checks do pure md02

| Arquivos | O que são | Sucesso |
|---|---|---|
| `eval_pure_seed0_md02_0` | avaliação do modelo md02, seed 0 (cópia oficial em `../baseline-oficial/`, com números ligeiramente diferentes) | 0.980 |
| `eval_pure_seed0_md02_re_0` | reavaliação do mesmo modelo, com `--success-bonus 5.0` | 0.900 |
| `eval_results_0` | mesmo modelo, rodado com `--render` (debug visual) | 0.980 |
| `eval_model0_target0_modepure_0` | removido por ser idêntico a `../baseline-oficial/` (re-execução com `--target-seed 0` e `--success-bonus 5.0`) | 0.980 |
| `eval_model{1..4}_target*_modepure_*` | pure md02, seeds 1 a 4, `target_seed = seed`; movidos para a raiz de `results/` (lidos pelo `plot_eval.py`) | 0.725 / 0.875 / 0.97 / 0.72 |

Observações:

- O `--success-bonus` na avaliação só altera o retorno, não a taxa de sucesso.
  A diferença 0.98 x 0.90 entre as duas avaliações do mesmo modelo vem
  provavelmente da sequência de alvos: nelas `--target-seed` não foi passado
  (alvos sorteados do gerador global, diferentes a cada execução). Isso indica
  a variabilidade de avaliar com alvos não fixados.
- O baseline oficial da seed 0 está em `results/baseline-oficial/`.
- Cópias idênticas de avaliações que existiam também na raiz de `results/` e em
  `baseline-oficial/` foram removidas; as referências acima apontam para o local atual.

## 3. Check de sanidade do pirl com `alpha_tau = 0`

Modelos treinados com `--alpha-tau 0.0` (run-tag `md02_at0`), isto é, pirl sem
penalização, que deveria se comportar como o pure.

| Arquivos | Seed | Sucesso | Torque médio | Energia média |
|---|---|---|---|---|
| `eval_pirl_seed0_md02_at0_0` | 0 | 0.980 | 2.02 | 1.54 |
| `eval_pirl_seed0_md02_at0_re_0` | 0 (reavaliação) | 0.900 | 1.53 | 1.46 |
| `eval_pirl_seed1_md02_at0_1` | 1 | 0.910 | 1.41 | 1.31 |

Na seed 0 os números são idênticos aos do pure md02 (original e reavaliação),
como esperado. Na seed 1 o pirl com `alpha_tau=0` deu 0.91, contra 0.725 do
pure md02 da seed 1 na seção 2; essa diferença deve ter a mesma origem
(alvos não fixados na avaliação) mais a variação do treino.

## 4. Arquivos sem resultado

- `eval_seed0_md02_at0_0_cmd.txt` e `eval_seed1_md02_at0_1_cmd.txt` só têm o
  comando (`--mode both`, run-tag `md02_at0`), sem CSV nem summary. Provavelmente
  a execução falhou, já que não existe um modelo pure com esse tag.
