"""
run_coffee.py — 커피부스 씬 (임시: 프레임만 배치)

현재 단계: 로봇팔/로봇다이(스탠드)/컴프레셔를 모두 비우고, 임시 커피부스 프레임
(coffee_booth_frame)만 씬 중심에 세워 형상을 확인하는 뷰어.

  · 프레임: footprint 1.4 x 2.0 m, 높이 0.8 m, 기둥 50x50mm 4개 + 상단 레일 4개
  · 원점 = 바닥-중앙 → Vec3(0,0,0) 스폰이면 바닥(z=0)에 딱 얹힘
  · 창을 닫으면 종료. (아직 로봇/제어 없음)

실행:  conda activate issac_env  &&  python run_coffee.py
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil   # torch보다 먼저

from asset_config import get_asset_root
asset_root = get_asset_root()   # 컴퓨터마다 에셋 위치 자동 탐색/저장 (asset_config.py 참고)

# ============================================================ [1] 시뮬
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Coffee booth scene (frame only)")
sp = gymapi.SimParams()
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sp.dt = 1.0 / 60.0
sp.physx.solver_type = 1
sp.physx.use_gpu = True
sp.physx.num_position_iterations = 8
sp.physx.num_velocity_iterations = 1
sp.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, pp)

# ============================================================ [2] 씬
env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0), gymapi.Vec3(2.5, 2.5, 2), 1)

# 임시 커피부스 프레임 — 씬 중심(0,0,0). 원점이 바닥-중앙이라 z=0이면 바닥에 안착.
frame_opts = gymapi.AssetOptions(); frame_opts.fix_base_link = True
frame_asset = gym.load_asset(sim, asset_root, "urdf/coffee_booth_frame/coffee_booth_frame.urdf", frame_opts)
gym.create_actor(env, frame_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0)), "coffee_booth_frame", 0, 0)

gym.prepare_sim(sim)

# ============================================================ [3] 뷰어
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(2.6, -2.6, 1.8), gymapi.Vec3(0, 0, 0.4))

print("""
========== 커피부스 씬 (임시: 프레임만) ==========
 프레임: 1.4 x 2.0 m, 높이 0.8 m (기둥 50x50 x4 + 상단 레일)
 (로봇/제어 없음 — 창 닫으면 종료)
================================================""")

# ============================================================ [4] 루프
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
