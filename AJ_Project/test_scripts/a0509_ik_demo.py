"""
a0509_ik_demo.py
두산 A0509를 '좌표(x,y,z) 추종'으로 움직이는 데모.

기존 a0509_control_demo.py 는 관절각을 직접 줬지만,
이 데모는 목표 '좌표'를 주면 ikpy(역기구학)가 관절각으로 변환해서 팔 손끝(link_6)이
그 좌표를 따라가게 한다.

  좌표 웨이포인트 → [ikpy IK] → 6관절 목표각 → [Isaac Gym 위치제어(PD)] → 팔 이동

사용법:
  conda activate issac_env
  python a0509_ik_demo.py              # 뷰어로 보기
  python a0509_ik_demo.py --headless   # 화면 없이 (좌표 추적 오차만 출력)

의존: ikpy (pip install ikpy==3.3.4)
주의: ISAAC_ASSETS 환경변수로 에셋 경로 지정 (미설정 시 기본 경로 사용)
"""
import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from ikpy.chain import Chain

# ----- 튜닝 파라미터 -----
STIFFNESS = 600.0    # P게인
DAMPING   = 50.0     # D게인
HOLD_SEC  = 2.5      # 각 좌표 웨이포인트를 유지하는 시간(초)

asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")
asset_file = "urdf/doosan_a0509/a0509.urdf"
urdf_path  = os.path.join(asset_root, asset_file)

# ----- IK 체인 준비 (base_link → joint_1 팔 체인만; 'base' 더미 가지는 제외) -----
# active_links_mask: [Base(fixed)=False, joint_1..joint_6 = True] → 실제 6관절만 IK 대상
ik_chain = Chain.from_urdf_file(
    urdf_path,
    base_elements=["base_link", "joint_1"],
    active_links_mask=[False, True, True, True, True, True, True],
    name="a0509",
)

def ik_solve(target_xyz, seed_6=None):
    """목표 좌표(x,y,z) → 6관절 각도(라디안) 배열. seed_6: 이전 해(연속성용)."""
    init = np.zeros(len(ik_chain.links))
    if seed_6 is not None:
        init[1:7] = seed_6
    sol = ik_chain.inverse_kinematics(target_position=target_xyz, initial_position=init)
    reached = ik_chain.forward_kinematics(sol)[:3, 3]
    err_mm = np.linalg.norm(np.array(target_xyz) - reached) * 1000.0
    return sol[1:7].astype(np.float32), reached, err_mm

# ----- 시뮬 초기화 -----
custom = [{"name": "--headless", "action": "store_true", "help": "뷰어 없이"}]
args = gymutil.parse_arguments(description="A0509 IK (Cartesian) demo",
                               custom_parameters=custom)

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

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
actor = gym.create_actor(env, asset, pose, "a0509", 0, 1)

# ----- DOF 위치제어 + PD 게인 -----
props = gym.get_actor_dof_properties(env, actor)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(STIFFNESS)
props["damping"].fill(DAMPING)
gym.set_actor_dof_properties(env, actor, props)
print(f"제어 모드=POS, stiffness={STIFFNESS}, damping={DAMPING}")

# ----- 목표 '좌표' 웨이포인트 (베이스 원점 기준, m) — 이 사이를 순회 -----
waypoints = [
    [0.35,  0.00, 0.55],   # 정면 앞
    [0.30,  0.25, 0.45],   # 오른쪽 앞 아래
    [0.30, -0.25, 0.45],   # 왼쪽 앞 아래
    [0.00,  0.35, 0.60],   # 오른쪽 위
    [0.40,  0.00, 0.70],   # 정면 높이
]

# 각 웨이포인트를 미리 IK로 풀어 관절각으로 변환 (연속성 위해 직전 해를 seed로)
solved = []
seed = None
print("\n=== 웨이포인트 IK 사전 계산 ===")
for wp in waypoints:
    q6, reached, err = ik_solve(wp, seed_6=seed)
    solved.append(q6)
    seed = q6
    print(f"  좌표 {wp} → q(deg)={np.round(np.rad2deg(q6),1)}  오차 {err:.2f}mm")

# ----- 뷰어 -----
viewer = None
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, env,
                              gymapi.Vec3(1.6, 1.6, 1.3), gymapi.Vec3(0, 0, 0.5))

# ----- 제어 루프 -----
steps_per_wp = int(HOLD_SEC / sim_params.dt)
step = 0
gym.set_actor_dof_position_targets(env, actor, solved[0])
print(f"\n[웨이포인트 0] 좌표 목표 {waypoints[0]}")

def current_tcp():
    """현재 관절각 → FK → 손끝(link_6) 좌표."""
    ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
    full = np.zeros(len(ik_chain.links)); full[1:7] = ds["pos"]
    return ik_chain.forward_kinematics(full)[:3, 3]

running = True
while running:
    if step % steps_per_wp == 0:
        wp_idx = (step // steps_per_wp) % len(waypoints)
        # 새 목표로 바꾸기 전에, 방금까지 '유지하던' 웨이포인트의 실제 도달 오차를 먼저 보고
        if step > 0:
            prev_idx = ((step // steps_per_wp) - 1) % len(waypoints)
            tgt_prev = np.array(waypoints[prev_idx])
            cur = current_tcp()
            print(f"[웨이포인트 {prev_idx} 도달] 목표={np.round(tgt_prev,3)}  "
                  f"실제손끝={np.round(cur,3)}  오차={np.linalg.norm(tgt_prev-cur)*1000:.1f}mm")
        # 다음 목표 좌표로 전환
        gym.set_actor_dof_position_targets(env, actor, solved[wp_idx])
        print(f"[웨이포인트 {wp_idx}] → 좌표 목표 {waypoints[wp_idx]}")

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if viewer is not None:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        if gym.query_viewer_has_closed(viewer):
            running = False
    else:
        if step > steps_per_wp * len(waypoints) * 2:
            running = False
    step += 1

if viewer is not None:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
