# Braço planar 2 DoF — experimento atual

Este projeto contém a versão atualizada do ambiente PyBullet/Gymnasium. Os experimentos legados estão preservados em outra branch e em outro venv.

## Arquivos

- `two_link_arm.urdf`: modelo físico com centros de massa e inércias corrigidos.
- `two_link_arm_env.py`: ambiente Gymnasium, física a 240 Hz e decisões a 30 Hz.
- `scripts/tests/test_env.py`: testes de integridade e baseline de cinemática inversa.
- `train_rl.py`: piloto PPO puro com logs e modelos em `runs/`.
- `eval_rl.py`: avaliação determinística, CSVs e resumos em `results/`.
- `scripts/plots/`: gráficos de treino e avaliação dos experimentos atuais.
- `requirements.txt`: dependências principais.

## Preparação (PowerShell)

Com o venv atual ativado:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python scripts/tests/test_env.py
python -c "from stable_baselines3.common.env_checker import check_env; from two_link_arm_env import TwoLinkArmEnv; e=TwoLinkArmEnv(); check_env(e); e.close(); print('SB3 CHECK PASSOU')"
python -m pip freeze > requirements-lock.txt
```

## Teste de integração (não é resultado científico)

```powershell
python train_rl.py --seed 0 --steps 4096 --run-tag smoke
python eval_rl.py --seed 0 --run-tag smoke --episodes 10
```

## Piloto RL puro

```powershell
python train_rl.py --seed 0 --steps 300000 --run-tag pilot
python train_rl.py --seed 1 --steps 300000 --run-tag pilot
python eval_rl.py --seed 0 --run-tag pilot --episodes 100 --target-seed 10000
python eval_rl.py --seed 1 --run-tag pilot --episodes 100 --target-seed 10000
```

Os scripts preservam os parâmetros físicos, registram configurações e não sobrescrevem silenciosamente modelos existentes. A avaliação usa alvos de teste comuns às políticas; seus CSVs não devem ser misturados aos do protocolo antigo.

## Gráficos de treino e avaliação

Execute os comandos a partir da raiz do projeto.

Após treinar:

```powershell
python scripts/plots/plot_train.py --run-tag pilot
```

Lê `runs/pure/seed_*_pilot/monitor.csv` e salva figuras em `figs/train/`.
A linha de cada seed representa uma média móvel por episódios, com janela padrão
de 30 episódios; os traços claros mostram os valores brutos.

Após avaliar:

```powershell
python scripts/plots/plot_eval.py --run-tag pilot
```

Lê `results/eval_*_pilot.csv`, salva figuras em `figs/eval/` e a tabela
`figs/eval/resumo_por_seed.csv`. Para comparar as políticas de forma pareada,
use o mesmo `--target-seed` e o mesmo número de episódios em todas as avaliações.

Para ajustar a janela móvel ou selecionar os arquivos de avaliação:

```powershell
python scripts/plots/plot_train.py --run-tag pilot --window 50
python scripts/plots/plot_eval.py --pattern 'results/eval_pure_seed*_pilot.csv'
```

## Cuidados de interpretação

- O `Monitor` registra métricas por episódio, com retorno e contagem de decisões.
- `torque_abs_integral`, `energy_abs` e as taxas de excedência dos envelopes e de
  saturação usam os passos físicos, conforme `two_link_arm_env.py`.
- Episódios de uma mesma seed de treinamento não substituem seeds independentes;
  a variabilidade entre episódios difere da variabilidade entre treinos.
- Os CSVs de avaliação atuais não contêm trajetórias temporais completas.
  Curvas de torque ou velocidade ao longo do episódio exigem um coletor de
  trajetórias no avaliador.
- Com 0% de sucesso, os gráficos de esforço e distância ajudam no diagnóstico,
  mas não representam eficiência de trajetórias bem-sucedidas.
