"""Valida propriedades inerciais e cinemática de two_link_arm_v2.urdf.
Execute: python validate_urdf_v2.py
Requer: pybullet
"""
from pathlib import Path
import math
import pybullet as p

urdf = str(Path(__file__).with_name('two_link_arm_v2.urdf'))
client = p.connect(p.DIRECT)
try:
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    robot = p.loadURDF(
        urdf, basePosition=[0, 0, 0], useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client,
    )
    assert p.getNumJoints(robot, physicsClientId=client) == 2
    expected = [
        (0.20, (0.000083333333, 0.004208333333, 0.004208333333)),
        (0.15, (0.000040000000, 0.003145000000, 0.003145000000)),
    ]
    for j, (mass, inertia) in enumerate(expected):
        dyn = p.getDynamicsInfo(robot, j, physicsClientId=client)
        print(f'link{j+1}: massa={dyn[0]:.6f} kg | inércia={dyn[2]} kg.m² | CM local={dyn[3]} m')
        assert math.isclose(dyn[0], mass, abs_tol=1e-8)
        for got, want in zip(dyn[2], inertia):
            assert math.isclose(got, want, abs_tol=1e-6)
        assert all(math.isclose(a, b, abs_tol=1e-6) for a,b in zip(dyn[3], (0.25,0,0)))

    # Validar ponta por geometria de um braço planar de dois elos.
    for q1, q2 in [(0.0,0.0), (0.5,-0.8), (-1.2,1.5)]:
        p.resetJointState(robot, 0, q1, physicsClientId=client)
        p.resetJointState(robot, 1, q2, physicsClientId=client)
        state = p.getLinkState(robot, 1, computeForwardKinematics=True, physicsClientId=client)
        tip, _ = p.multiplyTransforms(state[4], state[5], [0.5,0,0], [0,0,0,1])
        expected_xy = (0.5*math.cos(q1)+0.5*math.cos(q1+q2),
                       0.5*math.sin(q1)+0.5*math.sin(q1+q2))
        err = math.dist(tip[:2], expected_xy)
        print(f'q=({q1:+.2f}, {q2:+.2f}) | ponta={tip[:2]} | erro={err:.3e} m')
        assert err < 1e-5
    print('VALIDAÇÃO BÁSICA APROVADA')
finally:
    p.disconnect(physicsClientId=client)
