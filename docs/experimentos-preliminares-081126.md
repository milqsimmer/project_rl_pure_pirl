# Experimentos preliminares — braço 2-DOF, RL puro x PIRL

Relatório de 08/11/2026 sobre os experimentos feitos até aqui (branches `v4-nova-abordagem` e
`v5-penalidades-envelope`). Segue quatro seções: configuração original, experimentos
realizados, resultados e limitações identificadas. Todos os números vêm dos CSV/JSON
versionados em `results/`, `tabelas/` e `figs/`; o anexo lista onde cada conjunto está.

## Resumo

- **Baseline estável:** com `max_delta_theta = 0.2`, `lambda_a = 0.01` e bônus de sucesso 5.0
  (config `md02`), o RL puro chega a ~0,78 de sucesso médio em 10 seeds.
- **O termo físico de torque do PIRL não produziu efeito detectável** nas métricas de segurança
  (velocidade, torque, ângulo) em nenhuma variante testada: por passo, terminal, com pesos
  pequenos ou grandes. Pesos grandes degradam o sucesso e levam a políticas degeneradas.
- **Causa principal identificada:** as ações da política ficam saturadas no limite do espaço de
  ação (média pré-clip 5 a 6 vezes acima do limite), então termos de reward que dependem do
  tamanho da ação quase não geram gradiente. Normalizar o espaço de ação para [−1, 1] corrige
  isso e permite que um termo de velocidade reduza o tempo acima do envelope, com custo em sucesso.
- **O torque é um pulso de aceleração**, proporcional à velocidade atingida e quase independente
  da duração do episódio. O torque médio por passo é uma métrica enganosa quando a duração muda.
- **Único efeito mensurável de segurança até agora:** penalização de velocidade (`beta_omega`)
  com ação normalizada reduz a fração de passos acima de 80% do limite de velocidade
  (97% → 87% a 73%), a custo de 22 a 36 pontos percentuais de sucesso (2 seeds).

---

## 1. Configuração original

### 1.1 Modelo físico (`two_link_arm.urdf`)

| Item | Valor |
|---|---|
| Estrutura | braço planar de 2 elos, em plano horizontal; base fixa (massa 0) |
| Elo 1 | comprimento 0,5 m, massa 0,2 kg, inércia `ixx = iyy = izz = 0,01` |
| Elo 2 | comprimento 0,5 m, massa 0,15 kg, inércia `ixx = iyy = izz = 0,01` |
| Juntas | revolutas, eixo `z` (vertical), limites ±3,14 rad, `effort = 20`, `velocity = 5` |
| Ponta (efetuador) | no fim do elo 2 (0,5 m da junta 2, ao longo do eixo do elo) |
| Simulação | PyBullet, passo de 1/240 s, um passo de física por passo do ambiente |
| Motores | `POSITION_CONTROL` com `force = 20` e ganhos padrão do PyBullet |

### 1.2 Tarefa

- Alvo no plano XY sorteado com x ∈ [0,1; 0,9] e y ∈ [−0,5; 0,5], aceito se o raio for ≤ 1,0 m.
- Estado inicial: θ₁ = θ₂ = 0 (braço esticado, ponta em x = 1,0).
- Sucesso: distância ponta–alvo < 0,05 m. O episódio termina no sucesso ou em 200 passos
  (`TimeLimit`), ou seja, 0,83 s de tempo simulado.
- `target_seed = seed`: pure e pirl com a mesma seed veem a mesma sequência de alvos.

### 1.3 Observação e ação

- **Observação** (4 valores): `[θ₁, θ₂, x_alvo, y_alvo]`. Não inclui velocidades nem torques.
- **Ação** (2 valores): incremento de ângulo por junta, `Δθ ∈ [−max_delta_theta, +max_delta_theta]`.
  O alvo do motor é `clip(θ + Δθ, ±π)`. Na config `md02`, `max_delta_theta = 0,2` rad por passo.
  Opcionalmente (`normalize_actions`), a política atua em [−1, 1] e o ambiente multiplica por
  `max_delta_theta`.

### 1.4 Recompensa

RL puro (`pure`):

```
r_t = − d_t − λ_a·‖a_t‖ + B · 1[d_t < 0,05]          λ_a = 0,01 ; B = 5,0
```

PIRL (`pirl`) soma termos físicos ao reward do pure:

| Termo | Fórmula | Parâmetro |
|---|---|---|
| Torque **por passo** (versão original e hoje o padrão) | `− α_τ·(|τ₁|+|τ₂|)` em todo passo | `alpha_tau` |
| Torque **terminal** (variante) | `− α_τ·torque_médio_do_episódio`, só no passo de sucesso | `alpha_tau` + `--alpha-tau-mode terminal` |
| Envelope de velocidade | `− β_ω·max_j max(0, |ω_j| − 4,0)/1,0` | `beta_omega` |
| Envelope de torque | `− β_τ·max_j max(0, |τ_j| − 16)/4` | `beta_tau` |
| Envelope de ângulo | `− β_θ·max_j max(0, |θ_j| − 2,512)/0,628` | `beta_theta` |

O envelope é 80% dos limites do URDF (4,0 rad/s, 16 N·m, 2,512 rad). Os pilotos anteriores à
decisão usaram 2,64 rad no ângulo (84%). Os `beta` valem 0 por padrão.

### 1.5 Algoritmo

PPO do Stable-Baselines3, política `MlpPolicy`, um ambiente: `learning_rate = 3e-4`,
`n_steps = 2048`, `batch_size = 256`, `n_epochs = 10`, `gamma = 0,99`, `gae_lambda = 0,95`,
`clip_range = 0,2`, 2.000.000 de passos por treino. Avaliação: 200 episódios com política
determinística, 200 passos no máximo, alvos fixados por `target_seed = seed`.

### 1.6 Configuração-base `md02`

`steps = 2e6`, `lr = 3e-4`, `lambda_a = 0,01`, `success_bonus = 5,0`, `max_delta_theta = 0,2`,
`max_steps = 200`. Foi escolhida a partir da calibração do `pure` na seed 0 (E1, seção 3.2).

### 1.7 Métricas de segurança medidas na avaliação

Por episódio, a partir dos valores por passo (`eval_rl.py`):

- torque: pico, torque total do episódio (`tau_sum_total`), torque médio por passo e fração de
  passos com `|τ|` acima de p% de 20 N·m;
- velocidade: pico de `|ω|` e fração de passos com `|ω|` acima de p% de 5 rad/s;
- ângulo: pico de `|θ|`, fração de passos acima de p% de 3,14 rad e fração de passos em que o
  alvo comandado (`θ + Δθ`) passa de 3,14;
- suavidade: jerk RMS (diferença finita de ω) e variação média da ação `‖Δa‖`.

Com p ∈ {70, 80, 90, 100}. As avaliações de md02 (10 seeds) são anteriores às frações por
percentual e têm as medidas "perto do limite" (90% da velocidade; ângulo a menos de 0,2 rad
do limite; torque ≥ 99%).

---

## 2. Experimentos realizados

| # | Experimento | Variável | Seeds | Local |
|---|---|---|---|---|
| E0 | Primeira rodada (pré-calibração) | `max_delta_theta` padrão, sem bônus | 0 a 4 | `results/md01/` |
| E1 | Calibração do `pure` | `lambda_a`, bônus, `max_delta_theta`, passos | 0 | `results/pure_hiperparametros/` |
| E2 | Baseline md02: pure x pirl com `alpha_tau = 2e-4` por passo | — | 1 a 10 | `results/md02/` |
| E3 | Varredura de `alpha_tau` por passo (valores baixos) | 2e-4, 3e-4, 5e-4 | 0 | `results/pirl_alpha_tau_por_passo/` |
| E4 | `alpha_tau` terminal | 0,01 a 0,05 (seeds 0 a 4), 0,1 a 0,5 (seed 0), 0,3 (seeds 1 a 10), 1,0 e 3,0 (seed 1) | vários | `results/pirl_alpha_tau_terminal/`, `results/md02_at03_term/`, `results/md02_at_sweep_seed1/` |
| E5 | Normalização das ações (`normalize_actions`) + controle pure | ação em [−1, 1] | 0 a 2 | `results/md02_nrm_pilot/`, `results/md02_alpha_seed0/` |
| E6 | Penalizações operacionais (`beta_omega`, `beta_tau`, `beta_theta`) | fator ×0,5, ×1, ×3 sobre (0,05; 0,05; 0,1) a (0,3; 0,3; 0,6) | 1, 2 | `results/md02_lim_pilot/`, `results/md02_nrm_pilot/` |
| E7 | `alpha_tau` por passo, valores altos, bruto e normalizado | 0,01, 0,03, 0,1 | 0 | `results/md02_alpha_seed0/` |
| E8 | `alpha_tau` por passo, valores baixos, normalizado | 0,001, 0,003 | 0 a 2 | `results/md02_alpha_nrm/` |

Análises: comparação pareada por seed (diferença e t de Student pareado) e sensibilidade ao
envelope em 70/80/90/100% (`scripts/graficos/plot_sensibilidade.py`).

### 2.1 Diagnósticos feitos fora dos treinos
- resposta do ambiente a ação constante (relação ação–velocidade–torque);
- média pré-clip da política (quão saturada a ação está);
- trajetória de episódios dos modelos colapsados.

### 2.2 Notas de método
- Pure e pirl sempre são comparados na mesma seed e com a mesma sequência de alvos.
- Com normalização, o controle correto é o pure **também normalizado**; a normalização muda o
  comportamento dos dois modos.
- Os testes com 2 a 3 seeds têm baixo poder estatístico (para n = 3, |t| > 4,30 seria p < 0,05):
  só efeitos grandes seriam detectáveis.

---

## 3. Resultados

### 3.1 E0 — primeira rodada
Sucesso entre 0,29 e 0,64 nos dois modos, sem vantagem consistente do pirl. Motivou a calibração.

### 3.2 E1 — calibração do `pure` (seed 0)

| Variação (demais parâmetros iguais) | Sucesso |
|---|---|
| base (`max_delta_theta` 0,1) | 0,62 |
| `lambda_a` 0,005 | 0,445 |
| bônus de sucesso 8,0 | 0,42 |
| `max_delta_theta` 0,15 | 0,86 |
| **`max_delta_theta` 0,2 (md02)** | **0,98** |
| md02 com 0,5 / 1 / 1,5 M passos | 0,095 / 0,49 / 0,645 |

O ajuste que mais ajudou foi aumentar `max_delta_theta`. O `md02` precisa dos 2 M passos.
A avaliação da mesma configuração com alvos não fixados deu 0,90 (reavaliação) contra 0,98,
provavelmente por causa da sequência de alvos sorteada em cada avaliação (ver 4.5).

### 3.3 E2 — baseline md02, `alpha_tau = 2e-4` por passo (10 seeds, 2000 episódios por modo)

| Métrica | Pure | Pirl 2e-4 | t pareado |
|---|---|---|---|
| Sucesso | 0,777 | 0,769 | −0,12 |
| Distância final (m) | 0,134 | 0,132 | — |
| Torque médio por passo | 1,433 | 1,480 | +0,72 |
| Energia | 1,450 | 1,507 | +1,24 |
| Torque saturado (≥ 99%) | 2,69% | 3,52% | +2,01 |
| \|ω\| ≥ 90% do limite | 96,2% | 94,3% | −1,53 |
| Perto do limite articular | 5,5% | 4,9% | −0,33 |
| Comando além do limite articular | 5,3% | 4,5% | −0,39 |
| Jerk RMS (rad/s³) | 2,24e4 | 2,27e4 | +0,44 |

Nenhuma diferença passa de |t| = 2,26 (p < 0,05, n = 10). O pico de `|ω|` é 4,8 rad/s nos dois
modos em todos os episódios (nunca passa de 5). Os pesos 2e-4 a 5e-4 custam menos de 0,1 por
episódio, o que é desprezível.

### 3.4 E3 — `alpha_tau` por passo baixo (seed 0)

| `alpha_tau` | Sucesso | Torque médio |
|---|---|---|
| pure (referência) | 0,98 | 2,02 |
| 2e-4 | 0,835 | 1,51 |
| 3e-4 | 0,745 | 2,12 |
| 5e-4 | 0,655 | 1,51 |

O sucesso cai com `alpha_tau` sem queda consistente de torque.

### 3.5 E4 — `alpha_tau` terminal

- Pesos 0,01, 0,02 e 0,05 (seeds 0 a 4): com 0,01, sucesso médio 0,852 contra 0,854 do pure,
  torque médio 1,50 contra 1,63; valores maiores deram menos sucesso na seed 0.
- Seed 0 com 0,1 / 0,3 / 0,5: sucesso 0,795 / 0,965 / 0,980.
- **`alpha_tau = 0,3` em 10 seeds** (valor escolhido):

| Métrica | Pure | Pirl terminal 0,3 | t pareado |
|---|---|---|---|
| Sucesso | 0,777 | 0,823 | +0,64 |
| Torque médio | 1,433 | 1,467 | +0,52 |
| Energia | 1,450 | 1,479 | +0,66 |
| Torque saturado | 2,69% | 2,67% | −0,08 |
| \|ω\| ≥ 90% | 96,2% | 96,1% | −0,07 |
| Perto do limite articular | 5,5% | 4,1% | −0,66 |
| Comando além do limite | 5,3% | 3,9% | −0,65 |
| Jerk RMS | 2,24e4 | 2,27e4 | +0,37 |

Nenhuma métrica é significativa; os intervalos de confiança incluem zero. Variação entre seeds:
sucesso de 0,66 a 0,95 no pirl e de 0,38 a 0,97 no pure.

- **Pesos maiores (seed 1)**: `alpha_tau = 1,0`: sucesso 0,67, torque médio 1,41 (−7% contra o pure),
  limites iguais ao pure; `alpha_tau = 3,0`: sucesso 0,045 e duração média de 194 passos
  (colapso). Com 3,0 a penalização (~6,4 nos episódios concluídos) passa o bônus de sucesso (5,0),
  e como só existe quando há sucesso, falhar no tempo limite escapa dela.

### 3.6 E5 — normalização das ações
Com a ação em [−1, 1] a média pré-clip da política passa a ter escala útil. Nas seeds 1 e 2:

| | Sucesso | Torque total | Torque médio |
|---|---|---|---|
| pure sem normalizar | 0,80 | 129 | 1,53 |
| pure normalizado | 0,91 | 88 | 1,09 |

O pure normalizado também funciona melhor. Nas seeds 0 a 2 o pure normalizado tem sucesso
médio 0,84 (0,69, 0,87 e 0,96).

### 3.7 E6 — penalizações operacionais (`beta`), seeds 1 e 2, passos acima do envelope de 80%

Com ação normalizada (controle: pure normalizado):

| Modelo | Sucesso | Duração | Torque total | \|ω\| > 4,0 | \|τ\| > 16 | \|θ\| > 2,64 |
|---|---|---|---|---|---|---|
| pure normalizado | 0,91 | 97 | 88 | 96,7% | 1,8% | 2,8% |
| pirl ×0,5 | 0,69 | 119 | 88 | 86,6% | 1,7% | 6,8% |
| pirl ×1 | 0,56 | 132 | 89 | 72,5% | 1,5% | 5,3% |
| pirl ×3 | 0,18 | 183 | 45 | 1,0% | 0,4% | 1,7% |

Sem normalizar (controle: pure md02), a fração acima de 4,0 rad/s quase não muda
(95,8% → 94,6%, 92,9% e 85,4%) e o torque total não cai (129 → 121, 140 e 104).

- Com normalização, a média pré-clip da política passa de 1,04 (controle) para 0,68 e 0,48
  (×0,5 e ×1), contra um limite de 1,0.
- Com ×3 o braço se move devagar e quase não conclui a tarefa; as métricas de segurança
  "melhoram" porque a tarefa deixa de ser cumprida.
- Torque total só nos episódios com sucesso: 86 (controle), 78 (×0,5) e 84 (×1).
- A penalização de ângulo não reduziu a proximidade dos limites nos pesos moderados.

### 3.8 E7 — `alpha_tau` por passo alto (seed 0)

| Modelo | Sucesso | Torque total | \|ω\| > 4,0 | \|τ\| > 16 | \|θ\| > 2,51 |
|---|---|---|---|---|---|
| pure md02 (bruto) | 0,90 | 125 | 97,1% | 3,8% | 2,8% |
| pirl α = 0,01 | 0,57 | 124 | 89,1% | 3,6% | 16,2% |
| pirl α = 0,03 | 0,13 | 172 | 93,2% | 4,0% | 32,6% |
| pirl α = 0,1 | 0,12 | 99 | 77,9% | 1,3% | 33,2% |
| pure normalizado | 0,69 | 71 | 91,0% | 1,2% | 11,9% |
| pirl normalizado α = 0,01 | 0,20 | 150 | 90,7% | 1,2% | 30,0% |
| pirl normalizado α = 0,03 | 0,12 | 99 | 77,7% | 1,3% | 33,0% |
| pirl normalizado α = 0,1 | 0,12 | 99 | 77,9% | 1,3% | 33,2% |

A proximidade do limite articular piora de forma monotônica com `alpha_tau`.

### 3.9 E8 — `alpha_tau` por passo baixo, normalizado (seeds 0 a 2)

| Modelo | Sucesso | Torque total | \|ω\| > 4,0 | \|τ\| > 16 | \|θ\| > 2,51 |
|---|---|---|---|---|---|
| pure normalizado | 0,84 | 82 | 94,8% | 1,6% | 6,3% |
| pirl α = 0,001 | 0,76 | 89 | 93,2% | 1,7% | 7,9% |
| pirl α = 0,003 | 0,68 | 95 | 94,8% | 1,8% | 12,0% |

Diferenças contra o pure normalizado (t pareado, n = 3; |t| > 4,30 seria p < 0,05):
sucesso −0,08 (t = −0,88) e −0,16 (t = −1,14); torque total +7 (t = +0,52) e +13 (t = +1,01);
velocidade −1,6 e 0,0 p.p.; torque saturado +0,1 e +0,2 p.p.; ângulo +1,6 e +5,7 p.p. Nenhum
|t| passa de 1,4. O torque total do pure normalizado varia de 64 a 112 conforme a seed,
mais que qualquer efeito do `alpha_tau`.

### 3.10 Comportamento nos limites: política degenerada com `alpha_tau` alto
Com `alpha_tau` = 0,1 por passo (seed 0, sem normalizar), a política passa a dar a mesma ação em
qualquer estado, `(−0,2; +0,2)`, e ignora o alvo: ambos os ângulos terminam em −3,14 e +3,14
(braço dobrado até os batentes), com sucesso 0,12 e duração média de 185 passos. Com `alpha_tau` = 0,01
já aparece a junta 2 em 3,14 nos episódios que falham. O torque medido junto aos batentes é pequeno,
então estacionar nos limites minimiza a penalização por passo.

### 3.11 Torque e velocidade: relação com a ação (diagnóstico do ambiente)
Ação constante na junta 1:

| Ação (rad/passo) | \|ω\| (rad/s) | Pico de \|τ\| | Torque total em 40 passos | Nos 3 primeiros passos |
|---|---|---|---|---|
| 0,02 | 0,48 | 5,5 | 5,6 | 5,5 |
| 0,05 | 1,20 | 13,8 | 14,0 | 13,9 |
| 0,10 | 2,40 | 21,4 | 28,1 | 27,7 |
| 0,20 | 4,80 | 21,4 | 56,8 | 55,4 |

A velocidade é proporcional à ação (ω ≈ 24·a). O torque é gasto quase todo na aceleração e vai
a perto de zero em regime (mediana 0,03 na política treinada). Em 40 episódios da seed 1, 55% do
torque total saiu nos 10 primeiros passos. O torque total cresce com a velocidade atingida e não com a duração.

---

## 4. Limitações identificadas

### 4.1 Centros de massa e dinâmica do URDF
- Os blocos `<inertial>` dos elos 1 e 2 **não têm `<origin>`**, então o centro de massa de cada
  elo fica na origem do elo (no eixo da junta), enquanto a geometria visual e de colisão se
  estende de 0 a 0,5 m (origem em 0,25 m). O centro de massa não coincide com o centro geométrico.
- A inércia `ixx = iyy = izz = 0,01` é a mesma nos dois elos, apesar de massas diferentes (0,2 e
  0,15 kg), e não corresponde a uma haste uniforme de 0,5 m (cerca de 0,0042 kg·m² em torno do
  centro de massa de um elo de 0,2 kg).
- Como os eixos são verticais e o plano é horizontal, a gravidade não gera torque nas juntas:
  o torque medido é praticamente só o de aceleração mais o do controlador. Isso é coerente com
  o torque aparecer como pulso, mas também significa que o problema físico é mais simples e menos
  realista do que um braço em plano vertical.
- Esse efeito sobre os resultados não foi quantificado. Vale revisar o URDF (centro de massa e
  inércias) antes de generalizar as conclusões para um robô real.

### 4.2 Escala temporal
- Um passo do ambiente é um passo de física de 1/240 s; 200 passos são 0,83 s de simulação. Com
  ω ≈ 24·a e `max_delta_theta = 0,2`, a velocidade máxima é 4,8 rad/s. Os episódios bem-sucedidos
  levam em média cerca de 86 a 96 passos (seed 0), perto da metade do orçamento de 200.
- Nos testes com velocidade reduzida (`beta_omega` forte), a duração média sobe de ~97 para 132 e 183
  passos e o sucesso cai; é provável que o limite de passos passe a ser o fator limitante (hipótese
  não testada).
- A velocidade resulta de ganho de posição do motor, passo de 240 Hz e tamanho da ação
  (compatível com o ganho padrão do PyBullet: 0,1 × 240 = 24); o limite de 5 rad/s do URDF
  não é aplicado pelo controlador.
- Um comando de 0,2 rad por passo a 240 Hz equivale a 48 rad/s, muito acima da velocidade
  atingida (4,8 rad/s), o que explica o pico de torque saturado no início de cada movimento.

### 4.3 Saturação de ações
- Sem normalização, a média pré-clip da política fica em mediana 0,9 (pure) e 1,2 (pirl com
  `beta`), contra um limite de 0,2; em 61% a 94% dos passos a média passa do limite, com desvio de
  0,05 a 0,29. O ambiente recebe a ação cortada em ±0,2, então o reward não depende da média e
  termos que dependem do tamanho da ação (velocidade, torque, ‖a‖) quase não geram gradiente.
- Em ~70% dos passos a ação está no limite (|a| ≥ 0,19); a variação de ação entre passos é ~0,003.
  O comportamento é quase liga-desliga. É por isso que a velocidade parece "travada" em 4,8 rad/s
  (24 × 0,2): é o limite da ação, não um teto do controlador.
- A normalização para [−1, 1] reduz o problema (média pré-clip passa a variar, mediana 0,48 a 1,04),
  mas muda também o comportamento do pure (melhor sucesso e menos torque), de modo que exige um
  controle normalizado em toda comparação.

### 4.4 Problemas da recompensa
1. **Torque terminal só no sucesso:** falhar no tempo limite escapa da penalização. Com
   `alpha_tau = 3,0` (custo ~6,4 contra bônus 5,0) a política aprende a não concluir (sucesso 0,045).
2. **Torque por passo premia estacionar nos batentes:** o torque medido junto aos limites é baixo,
   então valores altos levam à política constante `(−0,2; +0,2)` com ângulos em ±3,14 (3.10). O
   termo físico de torque pode piorar a proximidade dos limites articulares.
3. **Sinal pequeno diante do ruído:** o retorno médio é ~−37 com desvio ~28; `alpha_tau` de 2e-4 a
   3e-3 custa 0,03 a 0,4 por episódio, imperceptível para o PPO. O custo dominante do reward é a
   distância (−d por passo).
4. **Torque médio por passo é uma métrica enganosa:** é o total dividido pela duração e diminui
   sozinho quando os episódios alongam (ex.: `alpha_tau = 0,01`: média 1,53 → 1,21, mas total 125
   → 124 e, só nos sucessos, 1,62 → 1,59). Deve-se usar o torque total do episódio.
5. **Torque e velocidade são a mesma alavanca** (tamanho da ação). Nenhuma variante reduziu o
   torque total sem colapsar o sucesso; só caiu em ×3 (88 → 45, sucesso 0,18).
6. **O termo de ângulo não ajudou** nos pesos moderados, e o ×3 não foi informativo porque a
   tarefa quase não era cumprida.
7. **Escala relativa dos pesos não calibrada:** `beta_theta` é o dobro de `beta_omega` e
   `beta_tau` por escolha inicial, e os três escalam juntos; não foi possível separar o efeito de
   cada um.

### 4.5 Outras limitações e observações sobre os dados
- **A observação não contém velocidades nem torques**, então a política não "vê" o que está sendo penalizado.
- **O envelope de 80% é uma convenção** de folga de projeto, não um resultado dos dados; foi
  adotado em 80% para velocidade e torque e 80% (2,512 rad) para o ângulo, com análise de
  sensibilidade em 70/80/90/100%. Os pilotos antigos de ângulo usaram 2,64 rad (84%). Os
  limites do URDF são parâmetros de simulação.
- **Poucas seeds nos testes recentes** (1 a 3), com variação grande entre seeds (por exemplo,
  torque total do pure normalizado de 64 a 112).
- **Variabilidade da avaliação:** o mesmo modelo `pure` da seed 0 deu 0,98 e 0,90 em avaliações
  sem `target_seed` fixado (provavelmente por sequências de alvos diferentes); as comparações
  pure x pirl só valem com o mesmo `target_seed`.
- **O reward registrado na avaliação não é o do treino:** o `eval_rl.py` constrói o ambiente com
  `alpha_tau` padrão (0,001); a coluna `return` do pirl não é comparável ao reward de treino.
- **Dados perdidos:** o `monitor.csv` da seed 3 da run `md02_at03_term` foi sobrescrito por um
  arquivo de 12 linhas; o modelo está intacto (a curva de treino dessa seed não pode ser refeita
  sem retreinar).
- **Duas gerações de métricas:** as avaliações de md02 (10 seeds) têm as medidas "perto do limite"
  (90% de velocidade; ângulo a menos de 0,2 rad do limite; torque ≥ 99%) e não as frações por
  percentual (70/80/90/100%), que existem só nos pilotos recentes. Para o baseline de 10 seeds,
  a sensibilidade ao envelope exigiria reavaliação.

---

## Anexo: rastreabilidade

| Conjunto | Runs (`runs/`) | Resultados (`results/`) | Tabelas e figuras |
|---|---|---|---|
| E2 baseline md02 | `runs_pure/seed_*_md02`, `runs_pirl/seed_*_md02_at2e4` (seeds 1 a 10) | `md02/` | `tabelas/*.csv`, `figs/figs_resultados/md02/`, `figs/figs_treino/` |
| E4 terminal 0,3 | `runs_pirl/seed_*_md02_at03_term` | `md02_at03_term/` | `tabelas/md02_at03_term/`, `figs/figs_resultados/md02_at03_term/` |
| E4 terminal 1,0 e 3,0 | `runs_pirl/seed_1_md02_at10_term`, `..._at30_term` | `md02_at_sweep_seed1/` | — |
| E6 `beta` | `runs_pirl/seed_{1,2}_md02_lim_*` e `..._md02_nrm_lim_*`, `runs_pure/seed_{1,2}_md02_nrm` | `md02_lim_pilot/`, `md02_nrm_pilot/` | — |
| E7 `alpha_tau` alto | `runs_pirl/seed_0_md02_ps_*`, `runs_pirl/seed_0_md02_nrm_ps_*`, `runs_pure/seed_0_md02_nrm` | `md02_alpha_seed0/` | `tabelas/md02_alpha_seed0/`, `figs/figs_resultados/md02_alpha_seed0/` |
| E8 `alpha_tau` baixo, normalizado | `runs_pirl/seed_{0,1,2}_md02_nrm_ps_a000{1,3}` | `md02_alpha_nrm/` | `tabelas/md02_alpha_nrm/`, `figs/figs_resultados/md02_alpha_nrm/` |

Comandos de treino e avaliação usados em cada run estão em `train_cmd.txt` (em cada pasta de run)
e nos `*_cmd.txt` de `results/`. Código: `two_link_arm_env.py`, `train_rl.py`, `eval_rl.py`,
`scripts/graficos/plot_eval.py`, `plot_train.py`, `plot_sensibilidade.py`.
