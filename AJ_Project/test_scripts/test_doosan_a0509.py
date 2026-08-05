"""
test_doosan_a0509.py
두산 A0509 URDF가 Isaac Gym에서 잘 로드되는지 확인하는 테스트.

사용법:
  # 1) 먼저 뷰어 없이 로드/파싱만 확인 (가볍고 빠름, .dae 메시 문제 조기 발견)
  python test_doosan_a0509.py --headless

  # 2) 잘 되면 뷰어로 실제 로봇 보기
  python test_doosan_a0509.py

주의: 반드시 delivery 환경 + unset PYTHONPATH 상태에서 실행.
"""

import os
from isaacgym import gymapi, gymutil   # isaacgym 을 torch 보다 먼저 import

# --- asset 경로 (프로젝트 공용 asset 폴더 기준) ---
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")
asset_file = "urdf/doosan_a0509/a0509.urdf"

# --- 인자 파싱 (--headless 는 커스텀으로 추가해야 함) ---
custom_parameters = [
    {"name": "--headless", "action": "store_true", "help": "뷰어 없이 로드/파싱만"},
]
args = gymutil.parse_arguments(description="Doosan A0509 load test",
                               custom_parameters=custom_parameters)

# --- 시뮬 초기화 ---
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)
if sim is None:
    raise RuntimeError("create_sim 실패")

# 바닥
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# --- A0509 asset 로드 ---
print(f"\n[LOAD] asset_root = {asset_root}")
print(f"[LOAD] asset_file = {asset_file}")
full = os.path.join(asset_root, asset_file)
print(f"[LOAD] 파일 존재? {os.path.isfile(full)}\n")

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True          # 팔 베이스 고정
asset_options.flip_visual_attachments = False
asset_options.armature = 0.01

a0509 = gym.load_asset(sim, asset_root, asset_file, asset_options)
if a0509 is None:
    raise RuntimeError("load_asset 실패 — URDF/메시 경로 확인 필요")

# --- 로드 정보 출력 ---
num_bodies = gym.get_asset_rigid_body_count(a0509)
num_dofs = gym.get_asset_dof_count(a0509)
print("=" * 50)
print(f"✅ A0509 로드 성공!")
print(f"   링크(rigid body) 수: {num_bodies}")
print(f"   관절(DOF) 수: {num_dofs}")
dof_names = gym.get_asset_dof_names(a0509)
print(f"   관절 이름: {dof_names}")
print("=" * 50)

# --- env 하나 만들고 actor 배치 ---
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
actor = gym.create_actor(env, a0509, pose, "a0509", 0, 1)

# --- 뷰어 (headless 아니면) ---
viewer = None
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, env,
                              gymapi.Vec3(1.5, 1.5, 1.2), gymapi.Vec3(0, 0, 0.4))

# --- 루프 ---
if args.headless:
    # 파싱/로드 확인만 — 몇 스텝만 돌리고 종료
    for _ in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    print("\n[HEADLESS] 로드/파싱 정상. 뷰어로 보려면 --headless 빼고 실행하세요.")
else:
    print("\n[VIEWER] 창을 닫으면 종료됩니다.")
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

# --- 정리 ---
if viewer is not None:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
