# Braço Robótico 2D com PyBullet, Gym e RL

Este projeto simula um braço robótico com dois graus de liberdade (2-DOF) utilizando PyBullet e um ambiente Gym customizado. A cinemática inversa é resolvida analiticamente e o braço é usado em experimentos de Aprendizado por Reforço (PPO) com duas variantes de recompensa: um baseline de RL puro e uma versão Physics-Informed (PIRL) com penalização de torque/energia.

## 🔧 Estrutura do Robô

O braço robótico é composto por:
- Um link base fixo
- Dois segmentos (`link1` e `link2`) conectados por juntas rotacionais
- Um alvo (esfera vermelha) no espaço 3D
- A ponta do `link2` é considerada o end-effector (atuador final)

## 📐 Cinemática Inversa Analítica

Ao invés de usar o solver interno de IK do PyBullet, a solução dos ângulos é feita manualmente via trigonometria:

- Utiliza a posição do alvo no plano XY
- Calcula os ângulos `θ1` e `θ2` que levam a ponta do braço até o alvo
- Resolve com base nas fórmulas clássicas de cinemática de braços planarem 2-DOF

## 🚀 O que já está implementado

- Ambiente Gym customizado (`TwoLinkArmEnv`) com métricas ricas no `info` (distância, torque, energia, sucesso).
- Resolução analítica de cinemática inversa para validação e geração de figuras.
- Treino de políticas PPO em dois modos de recompensa (`pure` e `pirl`).
- Avaliação sistemática dos modelos com métricas de sucesso, distância final, esforço em torque e energia.
- Scripts de análise para escalas típicas (distância/ação/torque/energia) e inspeção detalhada de episódios.

## 📂 Estrutura
```
├── two_link_arm_env.py      # Ambiente Gym customizado para o braço 2-DOF
├── train_rl.py              # Script de treino PPO (modes: pure, pirl)
├── eval_rl.py               # Script de avaliação dos modelos treinados
├── analyze_scales.py        # Analisa escalas típicas de dist/ação/torque/energia
├── inspect_episodes.py      # Inspeção detalhada de episódios e CSVs representativos
├── run_experiments_seeds.py # Pipeline de treino+avaliação para várias seeds
├── plot_arm_from_env.py     # Gera figura 2D estilizada do manipulador
├── scripts_test/
│   ├── demo_inicial.py      # Demo inicial em PyBullet (sem Gym/RL)
│   ├── test_env.py          # Teste de IK + TwoLinkArmEnv (distância ponta–alvo)
│   ├── test_tip_visual.py   # Debug visual da ponta/alvo com overlays
│   └── sanity_check.py      # Sanidade de Gym + NumPy
├── figs_env/                # Figuras do ambiente (layout, espaço de trabalho, etc.)
├── figs_resultados/         # Gráficos/resultados de avaliação
├── figs_treino/             # Curvas de treino (recompensa, etc.)
├── runs_pure/               # Saídas de treino PPO (modo "pure")
├── runs_pirl/               # Saídas de treino PPO (modo "pirl")
├── results/                 # CSVs/JSONs de avaliação gerados por eval_rl.py
│   └── script_graficos_v3.py# Gera figuras e tabelas (trade-off sucesso/energia, etc.)
├── two_link_arm.urdf        # Modelo URDF do braço
├── requirements.txt         # Dependências do projeto
├── requirements_legacy.txt  # Versão alternativa de dependências (legado)
├── AGENTS.md                # Guia para agentes de código
└── README.md                # Este arquivo
```

## ▶️ Como rodar

1. Crie um ambiente virtual (opcional, mas recomendado) e instale as dependências:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate

   pip install -r requirements.txt
   ```

2. Simulação / demonstração inicial (PyBullet puro, sem Gym/RL):
   ```bash
   python scripts_test/demo_inicial.py
   ```

3. Testes rápidos e debug do ambiente:
   - Sanidade de Gym + NumPy:
     ```bash
     python scripts_test/sanity_check.py
     ```
   - Teste de IK + ambiente (vários alvos aleatórios):
     ```bash
     python scripts_test/test_env.py
     ```
   - Debug visual da ponta e do alvo (overlays na GUI do PyBullet):
     ```bash
     python scripts_test/test_tip_visual.py
     ```

## 🧠 Requisitos
- Python 3.9+
- Gym 0.21 (API legada: `reset() -> obs`, `step() -> (obs, reward, done, info)`)
- PyBullet
- NumPy
- stable_baselines3 (PPO)


## Treino e avaliação de RL

### Treino (PPO)

1. Modo "pure" (distância + penalidade leve de ação):
   ```bash
   python train_rl.py --mode pure --seed 0 --steps 300000
   ```

2. Modo "pirl" (distância + ação + penalidade em torque):
   ```bash
   python train_rl.py --mode pirl --seed 0 --steps 300000
   ```

3. Exemplos de sweep de seeds (para experimentos mais completos):
   ```bash
   # pure
   python train_rl.py --mode pure --seed 0 --steps 300000
   python train_rl.py --mode pure --seed 1 --steps 300000
   python train_rl.py --mode pure --seed 2 --steps 300000
   python train_rl.py --mode pure --seed 3 --steps 300000
   python train_rl.py --mode pure --seed 4 --steps 300000

   # pirl
   python train_rl.py --mode pirl --seed 0 --steps 300000
   python train_rl.py --mode pirl --seed 1 --steps 300000
   python train_rl.py --mode pirl --seed 2 --steps 300000
   python train_rl.py --mode pirl --seed 3 --steps 300000
   python train_rl.py --mode pirl --seed 4 --steps 300000
   ```

4. Saídas dos treinos:
    - `runs_pure/seed_<seed>/monitor.csv` e `runs_pure/seed_<seed>/ppo_model_<seed>.zip`.
    - `runs_pirl/seed_<seed>/monitor.csv` e `runs_pirl/seed_<seed>/ppo_model_<seed>.zip`.


### Avaliação

1. Avaliar um único modo:
   ```bash
   python eval_rl.py --mode pure --episodes 20 --print-episodes --render
   ```
   ou
   ```bash
   python eval_rl.py --mode pirl --episodes 100
   ```

2. Comparar diretamente os dois modos:
   ```bash
   python eval_rl.py --mode both --episodes 100 --out results/eval_results --seed 0
   ```

3. Exemplo de avaliação com foco em torque/“esforço”:
   ```bash
   python eval_rl.py --mode both --episodes 100 --out results_torque.csv
   ```

4. Saídas das avaliações:
    - `results/eval_results_<seed>.csv` (métricas por episódio).
    - `results/eval_results_<seed>_summary.json` (resumo agregado em JSON).
    - Quando usar `run_experiments_seeds.py`, os arquivos têm o padrão
      `results/eval_official_seed<seed>_<seed>.csv` + JSON associado.


## 🔁 Experimentos com várias seeds

Para reproduzir um conjunto de experimentos de treino+avaliação para várias seeds:

```bash
python run_experiments_seeds.py --first-seed 0 --last-seed 4 \
    --steps 300000 --episodes 200
```

Esse script chama internamente `train_rl.py` (modos `pure` e `pirl`) e depois
`eval_rl.py --mode both` para cada seed, gerando arquivos
`results/eval_official_seed<seed>_<seed>.csv` e respectivos JSONs.


## 📊 Análises adicionais

- Escalas típicas de distância/ação/torque/energia:
  ```bash
  python analyze_scales.py --mode both --policy trained --seed 0 \
      --episodes 50 --max-steps 200
  ```

- Inspeção detalhada de episódios representativos (baixa/alta energia,
  sucesso/fracasso) e geração de CSVs passo a passo:
  ```bash
  python inspect_episodes.py --mode both --first-seed 0 --last-seed 4 \
      --episodes-per-config 20 --max-steps 200
  ```

  - Geração de gráficos e tabelas agregadas (sucesso, distância, esforço, energia)
  a partir dos CSVs de avaliação oficial:
  ```bash
  python results/script_graficos_v3.py
  ```


## 🔗 Exemplo de pipeline completo

Exemplo mínimo (single-seed) do fluxo treino → avaliação → análises → gráficos:

```bash
# 1) Treino PPO para os dois modos (seed 0)
python train_rl.py --mode pure --seed 0 --steps 300000
python train_rl.py --mode pirl --seed 0 --steps 300000

# 2) Avaliação conjunta (gera CSV/JSON em results/)
python eval_rl.py --mode both --seed 0 --episodes 100 \
    --out results/eval_results

# 3) Análises de escalas e episódios representativos
python analyze_scales.py --mode both --policy trained --seed 0 \
    --episodes 50 --max-steps 200
python inspect_episodes.py --mode both --first-seed 0 --last-seed 0 \
    --episodes-per-config 20 --max-steps 200

# 4) Geração de gráficos e tabelas agregadas
python results/script_graficos_v3.py
```


## Ajustes finos e dicas

- Se o braço mal se mexer durante o treino:
  - Aumente `force` no `POSITION_CONTROL` em `_apply_angles` dentro de `two_link_arm_env.py`,
    ou reduza a escala de ação (por exemplo, de ±0.1 para ±0.05).

- Se o movimento trepidar demais:
  - Reduza a escala de ação (ex.: ±0.05).
  - Aumente `max_episode_steps` no `TimeLimit` para 300 em `train_rl.py` (apenas para testes).

- Para “apimentar” o modo PIRL:
  - Adicione um termo extra de suavidade na recompensa, algo como
    `beta * (|Δθ1| + |Δθ2|)` por passo, se quiser penalizar variações bruscas de ângulo.
