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
├── two_link_arm_env.py       # Ambiente Gym customizado para o braço 2-DOF
├── train_rl.py               # Script de treino PPO (modes: pure, pirl)
├── eval_rl.py                # Script de avaliação dos modelos treinados
├── analyze_scales.py         # Analisa escalas típicas de dist/ação/torque/energia
├── inspect_episodes.py       # Inspeção detalhada de episódios e CSVs representativos
├── run_experiments_seeds.py  # Pipeline legado de treino+avaliação para várias seeds
├── run_train_target_seeds.py # Helper: treino pure/pirl com seed = target_seed
├── run_eval_target_seeds.py  # Helper: eval usando pares (model_seed, target_seed)
├── plot_train_compare.py     # Gráficos de treino (reward + sucesso, sucesso por seed)
├── plot_eval_compare.py      # Gráficos de eval (pure vs pirl por seed)
├── run_plots_train.py        # Gera todos os plots de treino a partir dos monitor.csv
├── run_plots_eval_all.py     # Gera todos os plots de eval (e opcionalmente escalas)
├── run_plots_results_paper.py# Pipeline de figuras/tabelas para a dissertação
├── plot_arm_from_env.py      # Gera figura 2D estilizada do manipulador
├── scripts_test/
│   ├── demo_inicial.py       # Demo inicial em PyBullet (sem Gym/RL)
│   ├── test_env.py           # Teste de IK + TwoLinkArmEnv (distância ponta–alvo)
│   ├── test_tip_visual.py    # Debug visual da ponta/alvo com overlays
│   └── sanity_check.py       # Sanidade de Gym + NumPy
├── scripts_graficos/         # Scripts extras para gráficos/tabelas (PDFs/LaTeX)
├── figs_env/                 # Figuras do ambiente (layout, espaço de trabalho, etc.)
├── figs_resultados/          # Gráficos/resultados de avaliação
├── figs_treino/              # Curvas de treino agregadas (script_plot_train)
├── runs_pure/                # Saídas de treino PPO (modo "pure")
├── runs_pirl/                # Saídas de treino PPO (modo "pirl")
├── results/                  # CSVs/JSONs de avaliação gerados por eval_rl.py
├── two_link_arm.urdf         # Modelo URDF do braço
├── requirements.txt          # Dependências do projeto
├── requirements_legacy.txt   # Versão alternativa de dependências (legado)
├── AGENTS.md                 # Guia para agentes de código
└── README.md                 # Este arquivo
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
   python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4
   ```

2. Modo "pirl" (distância + ação + penalidade em torque):
   ```bash
   python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4
   ```

3. Alinhando a sequência de alvos (pure vs pirl) com `target_seed`:
   ```bash
   python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4 --target-seed 0
   python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4 --target-seed 0
   ```

4. Exemplos de sweep de seeds (para experimentos mais completos):
   ```bash
   # pure
   python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pure --seed 1 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pure --seed 2 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pure --seed 3 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pure --seed 4 --steps 2000000 --learning-rate 3e-4

   # pirl
   python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pirl --seed 1 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pirl --seed 2 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pirl --seed 3 --steps 2000000 --learning-rate 3e-4
   python train_rl.py --mode pirl --seed 4 --steps 2000000 --learning-rate 3e-4
   ```

> Configuração baseline recomendada (pure), usada nos experimentos atuais:
> - `steps = 2_000_000`, `learning-rate = 3e-4`
> - `lambda_a = 0.01`
> - `success_bonus = 5.0`
> - `max-delta-theta = 0.2` (maior intervalo de ação ajuda a convergir)
> - `max-steps` no treino/avaliação = 200

4. Saídas dos treinos:
    - `runs_pure/seed_<seed>/monitor.csv` e `runs_pure/seed_<seed>/ppo_model_<seed>.zip`.
    - `runs_pirl/seed_<seed>/monitor.csv` e `runs_pirl/seed_<seed>/ppo_model_<seed>.zip`.


### Avaliação

1. Avaliar um único modo:
   ```bash
   python eval_rl.py --mode pure --seed 0 --episodes 200 --max-steps 200
   ```
   ou
   ```bash
   python eval_rl.py --mode pirl --seed 0 --episodes 200 --max-steps 200
   ```

2. Comparar diretamente os dois modos:
   ```bash
   python eval_rl.py --mode both --seed 0 --episodes 200 --max-steps 200 --out results/eval_official_seed0
   ```

3. Alinhar alvos de avaliação com o treino (`target_seed`):
   ```bash
   python eval_rl.py --mode both --seed 0 --episodes 200 --max-steps 200 --target-seed 0 \
       --out results/eval_official_seed0
   ```

4. Saídas das avaliações:
    - `results/eval_official_seed<seed>_<tag>.csv` (métricas por episódio).
    - `results/eval_official_seed<seed>_<tag>_summary.json` (resumo agregado em JSON).
    - Scripts de gráficos leem esses arquivos para gerar comparações pure vs pirl.


## 🔁 Experimentos com várias seeds

Para reproduzir um conjunto de experimentos de treino+avaliação para várias seeds:

Opção legado (simples):

```bash
python run_experiments_seeds.py --first-seed 0 --last-seed 4 \
    --steps 300000 --episodes 200
```

Opção recomendada (usando `target_seed` = `seed`):

```bash
# Treino pure + pirl para seeds 0..4
python run_train_target_seeds.py --first-seed 0 --last-seed 4 --steps 2000000

# Avaliação pure+pirl (usa os mesmos modelos e target_seeds)
python run_eval_target_seeds.py --mode both --model-first-seed 0 --model-last-seed 4
```


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
  python run_plots_train.py --first-seed 0 --last-seed 4
  python run_plots_eval_all.py --first-seed 0 --last-seed 4
  python run_plots_results_paper.py --first-seed 0 --last-seed 4 --include-inspected
  ```


## 🔗 Exemplo de pipeline completo

Exemplo mínimo (single-seed) do fluxo treino → avaliação → análises → gráficos:

```bash
# 1) Treino PPO para os dois modos (seed 0)
python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4
python train_rl.py --mode pirl --seed 0 --steps 2000000 --learning-rate 3e-4

# 2) Avaliação conjunta (gera CSV/JSON em results/)
python eval_rl.py --mode both --seed 0 --episodes 200 --max-steps 200 \
    --out results/eval_official_seed0

# 3) Análises de escalas e episódios representativos
python analyze_scales.py --mode both --policy trained --seed 0 \
    --episodes 50 --max-steps 200
python inspect_episodes.py --mode both --first-seed 0 --last-seed 0 \
    --episodes-per-config 20 --max-steps 200

# 4) Geração de gráficos e tabelas agregadas
python run_plots_results_paper.py --first-seed 0 --last-seed 0 --include-inspected
```


## Ajustes finos e dicas

- Se o braço mal se mexer durante o treino:
  - Primeiro tente aumentar `--max-delta-theta` em `train_rl.py` (por exemplo, de 0.1 para 0.15 ou 0.2).
  - Ajustar `force` em `_apply_angles` dentro de `two_link_arm_env.py` é possível, mas prefira manter o valor
    consistente com os experimentos de dissertação a menos que esteja apenas depurando o ambiente.

- Se o movimento trepidar demais:
  - Reduza `--max-delta-theta` (por exemplo, de 0.1 para 0.05).
  - Aumente `max_episode_steps` no `TimeLimit` para 300 em `train_rl.py` (apenas para testes).

- Para “apimentar” o modo PIRL:
  - Use os parâmetros existentes (`--alpha-tau` para torque, `--success-bonus` para reforçar sucesso) e,
    se for adicionar novos termos físicos, siga o plano descrito em `AGENTS.md` (primeiro estabilizar `pure`,
    depois incluir termos físicos de forma incremental, preferencialmente como termos terminais em energia,
    saturação ou postura).
