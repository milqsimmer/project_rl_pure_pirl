import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class TwoLinkArmEnv(gym.Env):
    def __init__(
        self,
        render: bool = False,
        reward_mode: str = "pure",
        lambda_a: float = 0.03,
        alpha_tau: float = 0.001,
        step_sleep: float = 0.0,
        max_delta_theta: float = 0.1,
        target_seed: int | None = None,
        success_bonus: float = 0.0,
        beta_omega: float = 0.0,
        beta_tau: float = 0.0,
        beta_theta: float = 0.0,
        normalize_actions: bool = False,
        omega_env: float = 4.0,
        tau_env: float = 16.0,
        theta_env: float = 2.512,  # 80% de 3.14 (pilotos antigos usaram 2.64)
        alpha_tau_mode: str = "per_step",
    ):
        super(TwoLinkArmEnv, self).__init__()
        self.reward_mode = reward_mode

        # pesos da função de recompensa
        self.lambda_a = float(lambda_a)
        self.alpha_tau = float(alpha_tau)

        # como alpha_tau age no modo pirl:
        #   "per_step": reward -= alpha_tau * (|tau1|+|tau2|) a cada passo (padrao)
        #   "terminal": reward -= alpha_tau * torque_medio_do_episodio, so no passo
        #               em que o alvo e atingido (usado pelas runs *_term)
        if alpha_tau_mode not in ("per_step", "terminal"):
            raise ValueError("alpha_tau_mode deve ser 'per_step' ou 'terminal'")
        self.alpha_tau_mode = alpha_tau_mode

        # penalizações por passo (só no modo pirl) sobre um envelope de operação
        # abaixo dos limites do URDF; todas com peso 0 por padrão (sem efeito).
        # Cada termo é um hinge normalizado: max(0, |x| - x_env) / (x_limite - x_env)
        self.beta_omega = float(beta_omega)
        self.beta_tau = float(beta_tau)
        self.beta_theta = float(beta_theta)
        self.omega_env, self.omega_limit = float(omega_env), 5.0  # rad/s
        self.tau_env, self.tau_limit = float(tau_env), 20.0  # N.m
        self.theta_env, self.theta_limit = float(theta_env), 3.14  # rad

        # bônus opcional aplicado quando alcança o alvo (dist < success_threshold)
        self.success_bonus = float(success_bonus)

        # pausa opcional após cada passo quando render=True (para visualização)
        self.step_sleep = float(step_sleep)

        # variação máxima de ângulo por passo (módulo da ação)
        self.max_delta_theta = float(max_delta_theta)

        # se True, a política atua em [-1, 1] e o ambiente escala por max_delta_theta
        self.normalize_actions = bool(normalize_actions)

        # gerador de números aleatórios para os alvos (independente do np.random global)
        if target_seed is None:
            self.target_rng = np.random
        else:
            self.target_rng = np.random.RandomState(int(target_seed))

        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # passo de tempo usado pelo motor fisico (para potencia/energia)
        params = p.getPhysicsEngineParameters()
        self.dt = float(params.get("fixedTimeStep", 1.0 / 240.0))

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("two_link_arm.urdf", basePosition=[0, 0, 0])

        self.link1 = 0
        self.link2 = 1

        self.l1 = 0.5
        self.l2 = 0.5

        self.offset_local = [self.l2, 0, 0]

        self.success_threshold = 0.05

        # Observação: [theta1, theta2, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -1.5, -1.5]),
            high=np.array([np.pi, np.pi, 1.5, 1.5]),
            dtype=np.float32,
        )

        # Ações: variação nos ângulos das juntas (incrementos).
        # Com normalize_actions=True o espaço da política é [-1, 1] e o ambiente
        # multiplica por max_delta_theta (prática recomendada pelo SB3); assim a
        # média da política não satura longe do limite e o reward consegue
        # empurrá-la. Com False (padrão) o espaço é [-max_delta_theta, +max_delta_theta].
        if self.normalize_actions:
            self.action_space = spaces.Box(
                low=-np.ones(2), high=np.ones(2), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-self.max_delta_theta, -self.max_delta_theta]),
                high=np.array([self.max_delta_theta, self.max_delta_theta]),
                dtype=np.float32,
            )

        self.target_pos = None
        self.state = None
        self.target_id = None

        # acumuladores por episódio (para métricas agregadas)
        self.ep_tau_sum_total = 0.0
        self.ep_energy = 0.0
        self.ep_steps = 0

        # índice do episódio (incrementado a cada reset)
        self.episode_idx = 0

        self.reset()

    def reset(self):
        self.theta1 = 0.0
        self.theta2 = 0.0

        self.target_pos = self._sample_target()

        # novo episódio
        self.episode_idx += 1

        # zera acumuladores por episódio
        self.ep_tau_sum_total = 0.0
        self.ep_energy = 0.0
        self.ep_steps = 0

        # coloca as juntas exatamente em (theta1, theta2) e zera velocidades
        p.resetJointState(self.robot, 0, self.theta1, targetVelocity=0.0)
        p.resetJointState(self.robot, 1, self.theta2, targetVelocity=0.0)

        # configura os motores de posição para manter esses ângulos
        self._apply_angles(self.theta1, self.theta2)

        self._create_target_visual()

        obs = self._get_obs()
        return obs

    def step(self, action):
        # ação em radianos por passo (desfaz a normalização, se ativa)
        action = np.asarray(action, dtype=np.float64)
        if self.normalize_actions:
            action = np.clip(action, -1.0, 1.0) * self.max_delta_theta

        # interpreta a ação como incremento em ângulo sobre o estado ATUAL
        # (estado atual é sempre sincronizado com o Bullet via getJointStates)
        th1_target = np.clip(self.theta1 + action[0], -np.pi, np.pi)
        th2_target = np.clip(self.theta2 + action[1], -np.pi, np.pi)

        # aplica ângulos alvo e avança simulação física
        self._apply_angles(th1_target, th2_target)
        p.stepSimulation()
        if self.render_mode and self.step_sleep > 0.0:
            time.sleep(self.step_sleep)

        # lê estado REAL das juntas após o passo
        js = p.getJointStates(self.robot, [0, 1])
        joint1_pos, joint1_vel, _, joint1_tau = js[0]
        joint2_pos, joint2_vel, _, joint2_tau = js[1]

        # sincroniza estado interno com o Bullet
        self.theta1 = float(joint1_pos)
        self.theta2 = float(joint2_pos)

        # distância ponta–alvo no plano (x, y)
        end_effector_pos = self._get_end_effector_pos()
        dist = np.linalg.norm(
            np.array(end_effector_pos[:2]) - np.array(self.target_pos)
        )

        # torques e velocidades físicos
        tau1 = abs(float(joint1_tau))
        tau2 = abs(float(joint2_tau))
        tau_sum = tau1 + tau2  # soma de |tau1| + |tau2|

        vel1 = float(joint1_vel)
        vel2 = float(joint2_vel)
        power = abs(tau1 * vel1) + abs(tau2 * vel2)

        # atualiza acumuladores por episódio
        if not np.isnan(tau_sum):
            self.ep_tau_sum_total += tau_sum
        if not np.isnan(power):
            self.ep_energy += power * self.dt
        self.ep_steps += 1

        if self.render_mode:
            print(f"[TwoLinkArmEnv] episodio={self.episode_idx} step={self.ep_steps}")

        # ====== REWARD ======
        # RL puro: -distância + penalidade leve de ação
        lam_a = self.lambda_a

        reward = -dist - lam_a * float(np.linalg.norm(action))

        if self.reward_mode == "pirl":
            # PIRL (padrao): penalizacao de torque a cada passo, proporcional a
            # |tau1| + |tau2|, em todo passo de todo episodio
            if (
                self.alpha_tau_mode == "per_step"
                and self.alpha_tau != 0.0
                and not np.isnan(tau_sum)
            ):
                reward -= self.alpha_tau * tau_sum

            # penalizacoes de envelope por passo (velocidade/torque/limite articular);
            # pesos beta = 0 por padrao (sem efeito)
            reward -= self._envelope_penalty(tau1, tau2, vel1, vel2)

        done = dist < self.success_threshold

        # bônus de sucesso: aplica apenas no passo em que alcança o alvo
        if done and self.success_bonus != 0.0:
            reward += self.success_bonus

        # modo "terminal": penalizacao de torque medio apenas no passo terminal (pirl)
        if (
            done
            and self.reward_mode == "pirl"
            and self.alpha_tau_mode == "terminal"
            and self.alpha_tau != 0.0
        ):
            mean_tau = (
                self.ep_tau_sum_total / self.ep_steps
                if self.ep_steps > 0
                else float("nan")
            )
            if not np.isnan(mean_tau):
                reward -= self.alpha_tau * mean_tau

        mean_tau = (
            self.ep_tau_sum_total / self.ep_steps if self.ep_steps > 0 else float("nan")
        )

        obs = self._get_obs()
        info = {
            "distance": float(dist),
            "final_distance": float(dist),
            "is_success": float(dist < self.success_threshold),
            "tau1": tau1,
            "tau2": tau2,
            "tau_sum": tau_sum,
            "omega1": vel1,
            "omega2": vel2,
            "power": power,
            # métricas acumuladas por episódio
            "tau_sum_total": float(self.ep_tau_sum_total),
            "episode_energy": float(self.ep_energy),
            "episode_mean_tau_sum": float(mean_tau),
        }

        return obs, reward, done, info

    @staticmethod
    def _hinge(x: float, x_env: float, x_limit: float) -> float:
        """max(0, |x| - x_env) / (x_limite - x_env)."""
        width = max(x_limit - x_env, 1e-9)
        return max(0.0, abs(float(x)) - x_env) / width

    def _envelope_penalty(self, tau1, tau2, vel1, vel2) -> float:
        """Penalização por passo por exceder o envelope de operação (pirl)."""
        pen = 0.0
        if self.beta_omega:
            w = max(self._hinge(vel1, self.omega_env, self.omega_limit),
                    self._hinge(vel2, self.omega_env, self.omega_limit))
            pen += self.beta_omega * w
        if self.beta_tau:
            t = max(self._hinge(tau1, self.tau_env, self.tau_limit),
                    self._hinge(tau2, self.tau_env, self.tau_limit))
            pen += self.beta_tau * t
        if self.beta_theta:
            th = max(self._hinge(self.theta1, self.theta_env, self.theta_limit),
                     self._hinge(self.theta2, self.theta_env, self.theta_limit))
            pen += self.beta_theta * th
        return float(pen) if not np.isnan(pen) else 0.0

    def _create_target_visual(self):
        if self.target_id is not None:
            p.removeBody(self.target_id)

        radius = 0.03
        red = [1, 0, 0, 1]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=red)
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=vis,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.1],
        )

    def _sample_target(self):
        while True:
            x = self.target_rng.uniform(0.1, 0.9)
            y = self.target_rng.uniform(-0.5, 0.5)
            if np.hypot(x, y) <= (self.l1 + self.l2):
                return [x, y]

    def _apply_angles(self, theta1, theta2):
        p.setJointMotorControl2(
            self.robot, 0, p.POSITION_CONTROL, targetPosition=theta1, force=20
        )
        p.setJointMotorControl2(
            self.robot, 1, p.POSITION_CONTROL, targetPosition=theta2, force=20
        )

    def _get_end_effector_pos(self):
        ls = p.getLinkState(self.robot, self.link2, computeForwardKinematics=True)
        link_pos = ls[4] if len(ls) > 4 else ls[0]
        link_ornt = ls[5] if len(ls) > 5 else ls[1]
        ee_pos, _ = p.multiplyTransforms(
            link_pos, link_ornt, self.offset_local, [0, 0, 0, 1]
        )
        return ee_pos

    def _get_obs(self):
        return np.array(
            [self.theta1, self.theta2, self.target_pos[0], self.target_pos[1]],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        if self.render_mode:
            time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect(self.physicsClient)
