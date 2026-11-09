"""Braço planar de 2 DoF: Gymnasium + PyBullet, versão experimental v2.

Design: PPO a 30 Hz, física a 240 Hz; incremento de referência persistente.
Sem treino PIRL aqui: recompensas iguais para comparação inicial futura.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data


class TwoLinkArmEnvV2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, urdf_path: str | Path | None = None,
                 render_mode: str | None = None, physics_hz: int = 240,
                 decision_hz: int = 30, episode_seconds: float = 2.0,
                 max_delta_theta: float = 0.1,
                 force_limit: float = 20.0, velocity_limit: float = 5.0,
                 lambda_action: float = 0.01, success_bonus: float = 5.0,
                 success_threshold: float = 0.05,
                 target_r_min: float = 0.2, target_r_max: float = 0.85):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode inválido")
        if physics_hz % decision_hz != 0 or decision_hz <= 0:
            raise ValueError("physics_hz precisa ser múltiplo de decision_hz")
        if not (0 < target_r_min < target_r_max < 1.0):
            raise ValueError("alvo deve estar dentro do alcance geométrico (raio < 1m)")
        self.render_mode = render_mode
        self.urdf_path = Path(urdf_path or Path(__file__).with_name('two_link_arm_v2.urdf')).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(self.urdf_path)
        self.physics_hz, self.decision_hz = int(physics_hz), int(decision_hz)
        self.frame_skip = physics_hz // decision_hz
        self.dt = 1.0 / physics_hz
        self.max_steps = round(episode_seconds * decision_hz)
        if self.max_steps < 1:
            raise ValueError("episódio deve ter pelo menos uma decisão")
        self.max_delta_theta = float(max_delta_theta)
        self.force_limit = float(force_limit)
        self.velocity_limit = float(velocity_limit)
        self.lambda_action = float(lambda_action)
        self.success_bonus = float(success_bonus)
        self.success_threshold = float(success_threshold)
        self.target_r_min, self.target_r_max = target_r_min, target_r_max
        self.q_limit = 3.14  # valor declarado no URDF
        self.q_envelope, self.omega_envelope, self.tau_envelope = 0.8 * self.q_limit, 4.0, 16.0
        self.l1 = self.l2 = 0.5
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # obs: q/pi [2], dq/vel_limit [2], target xy/1m [2], reference/pi [2]
        # velocidade observada pode superar o limite nominal: NÃO aplicar clip.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.client = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.client)
        self.robot = p.loadURDF(str(self.urdf_path), basePosition=[0,0,0],
                                useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE,
                                physicsClientId=self.client)
        if p.getNumJoints(self.robot, physicsClientId=self.client) != 2:
            raise ValueError('Esperadas exatamente duas juntas')
        self.target_id = None
        self.q_ref = np.zeros(2, dtype=np.float64)
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.n_steps = 0
        self._reset_counters()

    def _reset_counters(self):
        self.elapsed_physics_steps = 0
        self.energy_abs = 0.0
        self.torque_abs_integral = 0.0
        self.torque_sq_integral = 0.0
        self.torque_peak = np.zeros(2)
        self.velocity_peak = np.zeros(2)
        self.torque_envelope_steps = 0
        self.velocity_envelope_steps = 0
        self.position_envelope_steps = 0
        self.torque_saturation_steps = 0
        self.action_clip_decisions = 0
        self.reference_clip_decisions = 0
        self.safe_episode = True

    def _motor(self):
        for j in (0, 1):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=float(self.q_ref[j]),
                                    force=self.force_limit,
                                    maxVelocity=self.velocity_limit,
                                    physicsClientId=self.client)

    def joint_state(self):
        states = p.getJointStates(self.robot, [0, 1], physicsClientId=self.client)
        q = np.array([x[0] for x in states], dtype=np.float64)
        dq = np.array([x[1] for x in states], dtype=np.float64)
        tau = np.array([x[3] for x in states], dtype=np.float64)
        return q, dq, tau

    def tip_xy(self):
        ls = p.getLinkState(self.robot, 1, computeForwardKinematics=True,
                            physicsClientId=self.client)
        xyz, _ = p.multiplyTransforms(ls[4], ls[5], [self.l2, 0, 0], [0, 0, 0, 1])
        return np.asarray(xyz[:2], dtype=np.float64)

    def _obs(self):
        q, dq, _ = self.joint_state()
        obs = np.concatenate((q / math.pi, dq / self.velocity_limit,
                              self.target_xy, self.q_ref / math.pi)).astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError('observação não finita')
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_q = np.asarray(options.get('start_q', [0.0, 0.0]), dtype=np.float64)
        if start_q.shape != (2,) or np.any(np.abs(start_q) > self.q_limit):
            raise ValueError('start_q deve estar dentro dos limites articulares')
        if 'target_xy' in options:
            target = np.asarray(options['target_xy'], dtype=np.float64)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError('target_xy inválido')
            r = np.linalg.norm(target)
            if not (self.target_r_min <= r <= self.target_r_max) or target[0] < 0:
                raise ValueError('target_xy fora do domínio de alvos')
        else:
            r = np.sqrt(self.np_random.uniform(self.target_r_min ** 2, self.target_r_max ** 2))
            phi = self.np_random.uniform(-math.pi/2, math.pi/2)
            target = r * np.array([np.cos(phi), np.sin(phi)])
        self.target_xy = target
        self.q_ref = start_q.copy()
        self.n_steps = 0
        self._reset_counters()
        for j in (0, 1):
            p.resetJointState(self.robot, j, targetValue=float(start_q[j]),
                              targetVelocity=0, physicsClientId=self.client)
        self._motor()
        if self.target_id is not None:
            p.removeBody(self.target_id, physicsClientId=self.client)
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                    rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
        self.target_id = p.createMultiBody(baseVisualShapeIndex=shape,
                          basePosition=[float(target[0]), float(target[1]), 0.1],
                          physicsClientId=self.client)
        return self._obs(), {'target_xy': self.target_xy.tolist(), 'distance': float(np.linalg.norm(self.tip_xy() - target))}

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise ValueError('ação deve conter dois números finitos')
        bounded = np.clip(raw_action, -1.0, 1.0)
        self.action_clip_decisions += int(np.any(raw_action != bounded))
        raw_ref = self.q_ref + bounded * self.max_delta_theta
        new_ref = np.clip(raw_ref, -self.q_limit, self.q_limit)
        self.reference_clip_decisions += int(np.any(raw_ref != new_ref))
        self.q_ref = new_ref
        self._motor()
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client)
            q, dq, tau = self.joint_state()
            self.elapsed_physics_steps += 1
            self.torque_abs_integral += float(np.abs(tau).sum()) * self.dt
            self.torque_sq_integral += float(np.square(tau).sum()) * self.dt
            self.energy_abs += float(np.abs(tau * dq).sum()) * self.dt
            self.torque_peak = np.maximum(self.torque_peak, np.abs(tau))
            self.velocity_peak = np.maximum(self.velocity_peak, np.abs(dq))
            over_torque = np.any(np.abs(tau) > self.tau_envelope)
            over_velocity = np.any(np.abs(dq) > self.omega_envelope)
            near_joint = np.any(np.abs(q) > self.q_envelope)
            self.torque_envelope_steps += int(over_torque)
            self.velocity_envelope_steps += int(over_velocity)
            self.position_envelope_steps += int(near_joint)
            self.torque_saturation_steps += int(np.any(np.abs(tau) >= 0.99 * self.force_limit))
            self.safe_episode &= not (over_torque or over_velocity or near_joint)
        self.n_steps += 1
        q, dq, tau = self.joint_state()
        distance = float(np.linalg.norm(self.tip_xy() - self.target_xy))
        terminated = bool(distance < self.success_threshold)
        truncated = bool(self.n_steps >= self.max_steps and not terminated)
        reward = -distance - self.lambda_action * float(np.dot(bounded, bounded))
        if terminated:
            reward += self.success_bonus
        info = {
            'distance': distance, 'is_success': terminated,
            'target_xy': self.target_xy.tolist(), 'q_ref': self.q_ref.tolist(),
            'q': q.tolist(), 'dq': dq.tolist(), 'tau': tau.tolist(),
            'decision_steps': self.n_steps, 'physics_steps': self.elapsed_physics_steps,
            'elapsed_s': self.elapsed_physics_steps * self.dt,
            'torque_abs_integral': self.torque_abs_integral,
            'torque_sq_integral': self.torque_sq_integral,
            'energy_abs': self.energy_abs,
            'torque_peak': self.torque_peak.tolist(),
            'velocity_peak': self.velocity_peak.tolist(),
            'torque_envelope_rate': self.torque_envelope_steps / self.elapsed_physics_steps,
            'velocity_envelope_rate': self.velocity_envelope_steps / self.elapsed_physics_steps,
            'position_envelope_rate': self.position_envelope_steps / self.elapsed_physics_steps,
            'torque_saturation_rate': self.torque_saturation_steps / self.elapsed_physics_steps,
            'action_clip_decisions': self.action_clip_decisions,
            'reference_clip_decisions': self.reference_clip_decisions,
            'safe_episode': bool(self.safe_episode),
            'safe_success': bool(terminated and self.safe_episode),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        _, _, rgba, _, _ = p.getCameraImage(640, 480, physicsClientId=self.client)
        return np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]

    def close(self):
        if hasattr(self, 'client') and self.client is not None and p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)
            self.client = None
