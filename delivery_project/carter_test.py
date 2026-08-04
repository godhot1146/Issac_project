import os
import numpy as np
from isaacgym import gymapi, gymutil

# 1. 시뮬레이션 설정
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Carter 1.5m Shuttle")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# 2. 바닥 평면 설정
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 3. 환경 생성
env = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3), 1)

# 4. 에셋 로드 (4바퀴 URDF)
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/isaac_assets")
carter_asset = gym.load_asset(sim, asset_root, "urdf/carter/carter.urdf", gymapi.AssetOptions())

# 로봇 소환
carter_handle = gym.create_actor(env, carter_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.2)), "carter", 0, 1)

# 5. 제어 설정
dof_props = gym.get_actor_dof_properties(env, carter_handle)
dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
dof_props["stiffness"].fill(0.0)
dof_props["damping"].fill(800.0)
gym.set_actor_dof_properties(env, carter_handle, dof_props)

# 6. 왕복 주행 설정
target_a = 0.0
target_b = 1.5      # [변경] 3.0 -> 1.5
current_dir = 1     
speed = 3.0        

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(4.0, 4.0, 3.0), gymapi.Vec3(0.75, 0, 0))

print(f"1.5m 왕복 주행 모드 시작... ({target_a}m <-> {target_b}m)")

while not gym.query_viewer_has_closed(viewer):
    # 실시간 위치 피드백
    body_states = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
    curr_x = body_states['pose']['p']['x'][0]

    # 방향 전환 로직
    if current_dir == 1 and curr_x >= target_b:
        current_dir = -1
    elif current_dir == -1 and curr_x <= target_a:
        current_dir = 1

    # 4륜 속도 명령
    vel_cmd = speed * current_dir
    targets = np.array([vel_cmd, vel_cmd, vel_cmd, vel_cmd], dtype=np.float32)
    gym.set_actor_dof_velocity_targets(env, carter_handle, targets)

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
