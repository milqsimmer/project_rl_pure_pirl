# Braço planar 2 DoF — experimento atual

Este projeto contém a versão atualizada do ambiente PyBullet/Gymnasium. Os experimentos legados estão preservados em outra branch e em outro venv.

## Arquivos

- `two_link_arm.urdf`: modelo físico com centros de massa e inércias corrigidos.
- `two_link_arm_env.py`: ambiente Gymnasium, física a 240 Hz e decisões a 30 Hz.
- `test_env.py`: testes de integridade e baseline de cinemática inversa.
- `train_rl.py`: piloto PPO puro com logs e modelos em `runs/`.
- `eval_rl.py`: avaliação determinística, CSVs e resumos em `results/`.
- `requirements.txt`: dependências principais.

## Preparação (PowerShell)

Com o venv atual ativado:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python test_env.py
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
