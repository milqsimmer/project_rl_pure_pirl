"""Sanity checks independentes de PPO.

Execute da raiz: python scripts/tests/test_env.py.
"""

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from two_link_arm_env import TwoLinkArmEnv


def ik(x, y):
    l = 0.5
    c = np.clip((x * x + y * y - 2 * l * l) / (2 * l * l), -1.0, 1.0)
    q2 = math.acos(c)
    q1 = math.atan2(y, x) - math.atan2(l * math.sin(q2), l + l * math.cos(q2))
    return np.array([q1, q2])


def main():
    env = TwoLinkArmEnv()
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (8,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        target1 = np.asarray(info["target_xy"])
        obs2, info2 = env.reset(seed=123)
        assert np.allclose(target1, info2["target_xy"])
        print("PASSOU: API Gymnasium, observação, seed reproduzível")
        for q in ([0, 0], [0.5, -0.8], [-1.2, 1.5]):
            env.reset(options={"start_q": q, "target_xy": [0.5, 0.2]})
            x = 0.5 * math.cos(q[0]) + 0.5 * math.cos(q[0] + q[1])
            y = 0.5 * math.sin(q[0]) + 0.5 * math.sin(q[0] + q[1])
            assert np.linalg.norm(env.tip_xy() - [x, y]) < 1e-5
        print("PASSOU: cinemática direta")
        env.reset(seed=222)
        _, _, _, _, info = env.step(np.array([0.2, -0.4], np.float32))
        assert info["physics_steps"] == 8
        assert np.allclose(info["q_ref"], [0.02, -0.04], atol=1e-7)
        assert np.isfinite(info["energy_abs"])
        assert env.observation_space.contains(env._obs())
        print("PASSOU: 8 passos por decisão, referência persistente e métricas")
        env.reset(seed=321)
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step([0, 0])
            if terminated or truncated:
                break
        assert terminated or truncated
        assert info["physics_steps"] == info["decision_steps"] * 8
        print("PASSOU: término/truncamento e contadores")
        # Baseline de cinemática inversa. É diagnóstico, não meta garantida.
        successes = 0
        total = 30
        for i in range(total):
            _, inf = env.reset(seed=1000 + i)
            q_target = ik(*inf["target_xy"])
            for _ in range(env.max_steps):
                # q_ref é persistente; aproximar IK na taxa permitida por ação
                delta = (q_target - env.q_ref) / env.max_delta_theta
                action = np.clip(delta, -1.0, 1.0)
                _, _, term, trunc, inf = env.step(action)
                if term or trunc:
                    successes += int(term)
                    break
        print(f"Baseline IK de referência: {successes}/{total} episódios de sucesso")
        print("Não imponha meta de sucesso sem inspecionar trajetórias e alvos.")
        print("TESTES DE INTEGRIDADE APROVADOS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
