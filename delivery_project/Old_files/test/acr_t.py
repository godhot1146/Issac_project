import os
import math
import numpy as np
from isaacgym import gymapi, gymutil

# 1. 시뮬레이션 및 물리 엔진 초기화
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Franka-AMR Project")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
gym.set_light_parameters(sim, 0, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1.0, 0.0, -1.0))

# 2. 바닥 평면 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

# 3. 에셋 규격 및 가상 맵 정의
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = False       
env_opts.disable_gravity = False

floor_asset = gym.create_box(sim, 9.1, 9.1, 0.05, gymapi.AssetOptions()) 

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_opts.disable_gravity = False
tote_opts.armature = 0.05             
tote_opts.override_inertia = False   
tote_opts.override_com = False       
tote_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_VEL) 
tote_asset = gym.load_asset(sim, asset_root, "urdf/tote/whole/test.urdf", tote_opts)

# 4. 환경 생성 및 액터 배치
env = gym.create_env(sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 5), 1)
floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)

# 스폰 높이 0.4m 확보
robot_start_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.4))
tote_handle = gym.create_actor(env, tote_asset, robot_start_pose, "tote_asset", 0, 0)

# 5. 이름 기반 조인트 인덱스 찾기 및 제어 설정
dof_props = gym.get_actor_dof_properties(env, tote_handle)
num_dofs = len(dof_props)

dof_names = gym.get_actor_dof_names(env, tote_handle)
left_wheel_idx = -1
right_wheel_idx = -1

for i, name in enumerate(dof_names):
    if name == "left_wheel_joint":
        left_wheel_idx = i
    elif name == "right_wheel_joint":
        right_wheel_idx = i

if left_wheel_idx == -1 or right_wheel_idx == -1:
    left_wheel_idx = 0
    right_wheel_idx = 1

for i in range(num_dofs):
    dof_props['driveMode'][i] = int(gymapi.DOF_MODE_VEL)
    if i in [left_wheel_idx, right_wheel_idx]:
        dof_props['stiffness'][i] = 0.0
        dof_props['damping'][i] = 1000.0  
    else:
        dof_props['stiffness'][i] = 5000.0  
        dof_props['damping'][i] = 500.0    
    dof_props['velocity'][i] = 30.0     
    dof_props['effort'][i] = 1000.0      
gym.set_actor_dof_properties(env, tote_handle, dof_props)

# 바퀴 물리 사양
wheel_radius = 0.15   
wheel_base = 0.80     
wheel_velocities = np.zeros(num_dofs, dtype=np.float32)

# =================================================================
# [대폭 단순화] 6. 회전 없는 전/후진 왕복 변수 설정
# =================================================================
start_pos = None          # 이동 시작 지점 좌표
direction = 1.0           # 1.0 이면 전진, -1.0 이면 후진
distance_target = 2.0     # 목표 왕복 거리 2m
# =================================================================

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(4, 4, 4), gymapi.Vec3(0, 0, 0))

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 로봇 포즈 가져오기
    body_states = gym.get_actor_rigid_body_states(env, tote_handle, gymapi.STATE_POS)
    robot_pose = body_states['pose'][0] 
    current_pos = np.array([robot_pose['p']['x'], robot_pose['p']['y']])
    
    # 시작점 기준점 잡기
    if start_pos is None:
        start_pos = current_pos
        state_str = "직진(전진)" if direction > 0 else "직진(후진)"
        print(f"[상태] 2m {state_str} 시작")
        
    # 이동 거리 계산
    moved_distance = np.linalg.norm(current_pos - start_pos)
    
    # 2m에 도달하지 않았다면 계속 주행, 도달했다면 방향 전환
    if moved_distance < distance_target:
        linear_vel = 0.5 * direction  # direction이 -1이면 후진 속도가 됨
    else:
        linear_vel = 0.0
        start_pos = None              # 다음 턴의 시작점을 갱신하기 위해 초기화
        direction *= -1.0             # 부호를 반대로 뒤집음 (전진 <-> 후진)

    angular_vel = 0.0                 # 회전은 절대 하지 않음 고정

    # --- 차륜 역기학 계산 및 주입 ---
    left_wheel_vel = (linear_vel - (angular_vel * wheel_base / 2.0)) / wheel_radius
    right_wheel_vel = (linear_vel + (angular_vel * wheel_base / 2.0)) / wheel_radius
    
    # [수정] URDF 조인트 축 방향에 따른 바퀴 역전 현상 해결
    # 양수 속도를 주었을 때 우측 바퀴가 뒤로 돌아간다면 아래처럼 -를 붙여 방향을 맞춰줍니다.
    # 만약 반대로 돌면 left_wheel_vel 쪽에 -를 붙이거나 이 부분의 부호를 조절하시면 됩니다.
    right_wheel_vel = -right_wheel_vel 

    wheel_velocities.fill(0.0)
    wheel_velocities[left_wheel_idx] = left_wheel_vel
    wheel_velocities[right_wheel_idx] = right_wheel_vel
    
    gym.set_actor_dof_velocity_targets(env, tote_handle, wheel_velocities)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)