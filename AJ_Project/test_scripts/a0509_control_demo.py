"""
a0509_control_demo.py
두산 A0509를 위치 제어(DOF_MODE_POS + PD게인)로 움직이는 데모.
여러 목표 자세를 주기적으로 바꿔가며 팔이 따라가는 걸 보여준다.

사용법:
  python a0509_control_demo.py              # 뷰어로 보기
  python a0509_control_demo.py --headless   # 화면 없이 (각도 추적만 출력)

주의: delivery 환경 + unset PYTHONPATH 상태에서 실행.
"""

import os
import math
import numpy as np
from isaacgym import gymapi, gymutil

# ----- 튜닝 파라미터 (여기 값을 바꿔가며 실험) -----
STIFFNESS = 600.0   # P게인: 클수록 목표에 강하게 붙음 (너무 크면 진동/발산)
DAMPING   = 50.0    # D게인: 클수록 부드럽게 감쇠 (너무 작으면 오버슈트/떨림)
HOLD_SEC  = 2.0     # 각 목표 자세를 유지하는 시간(초)

asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")
asset_file = "urdf/doosan_a0509/a0509.urdf"

custom = [{"name": "--headless", "action": "store_true", "help": "뷰어 없이"}]
args = gymutil.parse_arguments(description="A0509 position control demo",
                               custom_parameters=custom)

# ----- 시뮬 초기화 -----
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# ----- A0509 로드 -----
opts = gymapi.AssetOptions()
opts.fix_base_link = True
opts.armature = 0.01
asset = gym.load_asset(sim, asset_root, asset_file, opts)
num_dofs = gym.get_asset_dof_count(asset)
print(f"A0509 로드: DOF {num_dofs}개")

# ----- env + actor -----
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
actor = gym.create_actor(env, asset, pose, "a0509", 0, 1)

# ----- 핵심: DOF 제어 모드 + PD 게인 설정 -----
props = gym.get_actor_dof_properties(env, actor)
props["driveMode"].fill(gymapi.DOF_MODE_POS)   # 위치 제어
props["stiffness"].fill(STIFFNESS)             # P게인
props["damping"].fill(DAMPING)                 # D게인
gym.set_actor_dof_properties(env, actor, props)
print(f"제어 모드=POS, stiffness={STIFFNESS}, damping={DAMPING}")

# ----- 목표 자세들 (6관절, 라디안) — 이 사이를 왔다갔다 -----
d = math.radians
poses = [
    np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),                     # 홈
    np.array([d(45), d(-30), d(60), 0, d(45), 0], dtype=np.float32),    # 자세 A
    np.array([d(-45), d(-60), d(90), d(30), d(-30), d(90)], dtype=np.float32),  # 자세 B
    np.array([0, d(-90), d(120), 0, d(60), 0], dtype=np.float32),       # 접힘
]

# ----- 뷰어 -----
viewer = None
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, env,
                              gymapi.Vec3(1.4, 1.4, 1.1), gymapi.Vec3(0, 0, 0.4))

# ----- 제어 루프 -----
steps_per_pose = int(HOLD_SEC / sim_params.dt)
step = 0
pose_idx = 0
gym.set_actor_dof_position_targets(env, actor, poses[pose_idx])
print(f"[목표 {pose_idx}] {np.round(poses[pose_idx], 2)}")

running = True
while running:
    # 일정 시간마다 다음 목표로 전환
    if step % steps_per_pose == 0:
        pose_idx = (step // steps_per_pose) % len(poses)
        gym.set_actor_dof_position_targets(env, actor, poses[pose_idx])
        # 현재 각도 읽어서 추적 상태 출력
        ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
        cur = np.round(ds["pos"], 2)
        print(f"[목표 {pose_idx}] target={np.round(poses[pose_idx], 2)}  현재={cur}")

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if viewer is not None:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        if gym.query_viewer_has_closed(viewer):
            running = False
    else:
        if step > steps_per_pose * len(poses) * 2:   # headless는 2바퀴 후 종료
            running = False

    step += 1

if viewer is not None:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
