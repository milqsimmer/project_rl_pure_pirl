# Braço planar v2 — validação antes do PPO

Este é o documento da versão atual. O `README.md` anterior permanece como
referência histórica; novas instruções e resultados entram neste arquivo.

Os arquivos atuais são `two_link_arm_env.py`, `two_link_arm.urdf` e
`requirements.txt`. Os testes e diagnósticos ficam em `scripts/tests/`.

* URDF corrigido com os centros de massa e inércias de caixas homogêneas.
* Gymnasium API: `reset() -> (obs, info)`; `step() -> (obs, reward, terminated, truncated, info)`.
* 240 Hz física, 30 Hz ação, 8 subpassos por decisão, 2 s máximo (60 decisões).
* Ação normalizada [-1,1] para cada junta, incremento de referência persistente ±0.1 rad por decisão.
* Observação de 8 dimensões: q/pi, dq/5, alvo XY (m), referência/pi.
* Recompensa apenas baseline: -distância - 0.01 * soma(a²) + 5 * sucesso.
* Penalizações PIRL **não** estão ativas: primeiro validar baseline e métricas.
* Sem `clip` das velocidades medidas; Box de observação sem limites artificiais.
* Métricas acumuladas em **cada passo físico**, inclusive integração de |torque|, torque² e potência absoluta.
* O envelope de 80% é uma hipótese experimental, não uma garantia de segurança física.

## Instalação (PowerShell, a partir da raiz do projeto)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/tests/validate_urdf.py
python scripts/tests/test_env.py
```

Para checagem automática de compatibilidade com SB3 (opcional):

```powershell
python -c "from stable_baselines3.common.env_checker import check_env; from two_link_arm_env import TwoLinkArmEnv; e=TwoLinkArmEnv(); check_env(e); e.close(); print('SB3 CHECK PASSOU')"
```

Os arquivos novos substituíram os anteriores e usam os nomes sem sufixo `v2`.
`train_rl.py`, `eval_rl.py` e os helpers da API antiga ainda aguardam migração;
os comandos do README anterior não representam o fluxo atual de treino.
Não iniciar treinos principais até a atualização desses scripts e a revisão das saídas.

Diagnósticos do controlador (execute da raiz):

```powershell
python scripts/tests/test_controller.py
python scripts/tests/test_controller_velocity.py
python scripts/tests/test_controller_actions.py
```

## Escolhas ainda abertas

- Comparar 30/60 Hz e horizontes usando alvos equivalentes em novo piloto.
- Verificar explicitamente desvios numéricos acima de 5 rad/s (maxVelocity não é barreira rígida).
- Avaliar baseline IK e decidir envelope operacional baseado em esforço viável.
- Tornar recompensa PIRL parametrizável **após** validar baseline.
