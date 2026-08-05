"""
run.py  (최소 구성 — 조립 전용)
두산 A0509 한 대를 바닥에 놓고 좌표 웨이포인트를 순회시킨다.

제어 로직(에셋 로드·스폰·DOF 제어·IK)은 전부 controllers/doosan_a0509_controller.py
안에 있고, 여기서는 "무대(sim·env) 준비 → 팔 배치 → 좌표 시퀀스 선언 → 루프"만 한다.

실행:
  conda activate issac_env
  cd ~/Desktop/Issac_project/AJ_Project
  python run.py
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil

# 제어 모듈은 controllers/ 폴더에 있으므로 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers"))
from doosan_a0509_controller import DoosanA0509Controller

# 에셋 경로 (다른 PC면 ISAAC_ASSETS 환경변수로 지정)
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")

HOLD_SEC = 2.5   # 각 좌표 웨이포인트를 유지하는 시간(초)

# ============================================================================
# [1] 시뮬레이션 / 물리 엔진 초기화  (sim = '우주', 물리 법칙이 적용되는 세계)
# ============================================================================
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="A0509 only - IK Cartesian control")

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

# 바닥면은 env가 아니라 sim에 붙음 (모든 env가 공유하는 무한 평면)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# ============================================================================
# [2] 무대(env) 준비 + 선반+로봇 통합 배치
#   선반과 로봇이 fixed joint로 결합된 통합 에셋(a0509_on_stand)을 하나로 스폰.
#   fix_base=False → 월드에 안 박고 통째로 바닥에 얹힘 (중력으로 안착).
#   IK는 순수 로봇팔 원본(a0509.urdf)으로 계산 → 좌표 웨이포인트는 로봇 베이스 기준.
# ============================================================================
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

arm = DoosanA0509Controller(
    gym, sim, env, asset_root,
    urdf="urdf/a0509_on_stand/a0509_on_stand.urdf",   # 액터: 선반+로봇 통합
    ik_urdf="urdf/doosan_a0509/a0509.urdf",         # IK 계산: 순수 로봇팔
    fix_base=False,                                  # 월드 비고정 → 바닥에 얹힘
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0, 0, 0)),
)
print(f"선반+A0509 통합 배치 완료 (바닥 안착, DOF {arm.num_dofs}개)")

# --- 에어 컴프레셔를 선반 내부(바닥판 위)에 배치 (STEP→URDF 변환 에셋) ---
comp_opts = gymapi.AssetOptions()
comp_opts.fix_base_link = False          # 고정 안 함 → 중력으로 선반 바닥판 위에 얹힘
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.12)), "air_compressor", 0, 0)
print("에어 컴프레셔 배치 완료 (선반 내부)")

# ============================================================================
# [3] 좌표 시퀀스 '선언' (무엇을 어디로 — 좌표만 나열)
#   좌표를 바꾸고 싶으면 이 리스트만 수정하면 된다.
# ============================================================================
waypoints = [
    [0.35,  0.00, 0.55],
    [0.30,  0.25, 0.45],
    [0.30, -0.25, 0.45],
    [0.00,  0.35, 0.60],
    [0.40,  0.00, 0.70],
]
solved = arm.plan_path(waypoints)   # 좌표 → 관절각 사전 IK (오차 출력)

# ============================================================================
# [4] 뷰어
# ============================================================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env,
                          gymapi.Vec3(1.6, 1.6, 1.3), gymapi.Vec3(0, 0, 0.5))

# ============================================================================
# [5] 메인 루프 (좌표 웨이포인트 순회)
# ============================================================================
steps_per_wp = int(HOLD_SEC / sim_params.dt)
arm.go_joints(solved[0])
print(f"\n[웨이포인트 0] → 좌표 목표 {waypoints[0]}")

step = 0
while not gym.query_viewer_has_closed(viewer):
    if step % steps_per_wp == 0:
        wp_idx = (step // steps_per_wp) % len(waypoints)
        # 직전 웨이포인트 실제 도달 오차 보고
        if step > 0:
            prev = ((step // steps_per_wp) - 1) % len(waypoints)
            tgt = np.array(waypoints[prev])
            cur = arm.current_tcp()
            print(f"[웨이포인트 {prev} 도달] 목표={np.round(tgt,3)}  "
                  f"실제손끝={np.round(cur,3)}  오차={np.linalg.norm(tgt-cur)*1000:.1f}mm")
        arm.go_joints(solved[wp_idx])
        print(f"[웨이포인트 {wp_idx}] → 좌표 목표 {waypoints[wp_idx]}")

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    step += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
