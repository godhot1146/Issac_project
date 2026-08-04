import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController

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

# =================================================================
# 3. [25평 공장 레이아웃 변형] 에셋 규격 및 가상 맵 정의
#    25평 규모(9.1m x 9.1m) 구조를 유지하되 내측 적재 랙은 모두 제거합니다.
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# 25평 정사각형 공장 사양 (9.1m * 9.1m)
room_size = 9.1          
wall_height = 10.0        # 공장 층고 (10m)
wall_thickness = 0.2     # 외벽 두께 (20cm)

floor_asset = gym.create_box(sim, room_size, room_size, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_size, wall_height, env_opts)
wall_y = gym.create_box(sim, room_size, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height, env_opts) # 코너 H빔 기둥

box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

vacuum_gripper_opts = gymapi.AssetOptions()
vacuum_gripper_opts.fix_base_link, vacuum_gripper_opts.flip_visual_attachments = True, True
vacuum_gripper_opts.armature = 0.01
vacuum_gripper_opts.thickness = 0.001
vacuum_gripper_opts.linear_damping = 0.0
vacuum_gripper_opts.angular_damping = 0.0
vacuum_gripper_opts.override_com = True
vacuum_gripper_opts.override_inertia = True

# 만약 텐서 API 제어를 위해 링크의 강체 관성이 필요하다면 아래 옵션을 켭니다.
vacuum_gripper_opts.override_com = True
vacuum_gripper_opts.override_inertia = True

vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120_box.urdf", vacuum_gripper_opts)

# =================================================================
# 4. 25평 오픈 팩토리 환경 생성 및 외벽/기둥 배치 (Actors)
# =================================================================
env = gym.create_env(sim, gymapi.Vec3(-room_size/2, -room_size/2, 0), gymapi.Vec3(room_size/2, room_size/2, room_size), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# 25평 외곽 경계벽 스폰
half_r = room_size / 2
w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_r, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_r, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_r, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_r, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
	gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

# 공장 프레임 유지를 위한 코너 사각 기둥 배치
p_offset = half_r - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset, p_offset]:
	for y in [-p_offset, p_offset]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# 외벽 보강 지지 기둥 정렬
pillar_side_positions = np.arange(-2.5, 3.5, 2.5)
for y_pos in pillar_side_positions:
	for x_pos in [-p_offset, p_offset]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)),f"side_pillar_{x_pos}_{y_pos}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# --- 로봇 인스턴스 스폰 ---
franka_handle = gym.create_actor(env, vacuum_gripper_asset, gymapi.Transform(p=gymapi.Vec3(-0.9, 0.0, 0.17)), "vacuum_gripper_asset", -1, 0)
pump_index = gym.find_actor_rigid_body_index(env, franka_handle, "cobot_pump_base", gymapi.DOMAIN_ACTOR)

if pump_index == -1:
    print("cobot_pump 링크를 찾을 수 없습니다. URDF 이름을 확인하세요.")
else:
    print(f"cobot_pump의 액터 내 인덱스: {pump_index}")

small_box_asset = gym.create_box(sim, 0.1, 0.1, 0.11, gymapi.AssetOptions())

# --- 컨트롤러 연동 ---
franka_ctrl = FrankaController(gym, sim, env, franka_handle, pump_link_name="cobot_pump_base", scale=1.2)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(2, 2, 1), gymapi.Vec3(1, 1, 0))

# 감지할 키 이벤트 등록 (예: SPACE 키를 "start_process"라는 이름의 이벤트로 등록)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")
# 필요하다면 다른 키도 추가 가능 (예: R 키를 리셋용으로 등록)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_process")

# 프로세스 제어를 위한 상태 플래그 변수 생성
is_active = False  # 스페이스바를 누르기 전까지는 로봇이 움직이지 않도록 보관

is_loaded = False
frame_count = 0

franka_ctrl.start_step = 0  
franka_ctrl.end_step = 1   
franka_ctrl.pause_duration = 5

robot_state = "MOVE_DOWN_1"    # 초기 상태: 첫 번째 하강
step_dur = 50                 # 각 스테이트별 구동 시간

# target_relative_xyz = [0.6436, -0.0028, -0.0148] # 1배 스케일링
target_relative_xyz = [0.7706, -0.0008, 0.0065] # 1.2배 스케일링

robot_base_pos = np.array([-0.9, 0.0])
box_pos = np.array([-0.3, 0.2])

# 베이스에서 박스를 바라보는 상대 벡터 (dx, dy)
delta_pos = box_pos - robot_base_pos

# 💡 1번 관절이 회전해야 할 정확한 라디안 각도 계산 (atan2 활용)
target_box_angle = np.arctan2(delta_pos[1], delta_pos[0])

target_position = np.array([0.5, 0.0, 0.8], dtype=np.float32)
target_orientation = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32) # 하강 자세 예시

robot_state_v2 = "MOVE_TO_TARGET"

while not gym.query_viewer_has_closed(viewer):
    gym.clear_lines(viewer)
	
    if robot_state_v2 == "MOVE_TO_TARGET":
        # 🔥 IK 함수 호출: 현재 상태를 기반으로 7자유도 목표 각도(9개 원소 배열)를 계산
        ik_targets = franka_ctrl.compute_ik_7dof(
            target_pos=target_position, 
            target_quat=target_orientation,
            max_iters=15, 
            tol=1e-3
        )
        
        if ik_targets is not None:
            # 💡 [핵심] 계산된 IK 목표치를 보간 버퍼(current_targets)에 주입합니다.
            # 이렇게 하면 마지막 2개 관절의 기존 타겟은 유지되면서 앞의 7개만 목적지가 바뀝니다.
            franka_ctrl.current_targets = ik_targets

        # 목표 지점에 근사했는지 체크 후 상태 전환
        ee_pos, _ = franka_ctrl.get_end_effector_pose()
        curr_p = np.array([ee_pos['x'], ee_pos['y'], ee_pos['z']])
        if np.linalg.norm(target_position - curr_p) < 0.01:
            print("[IK 완료] 목표 지점 도달!")
            robot_state_v2 = "NEXT_STATE"

    # 🎯 [중요] 보간 시스템과 물리 엔진은 매 틱마다 똑같이 수행되어야 합니다.
    franka_ctrl.update_joints_interpolation(dt=1.0/30.0)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 💡 [중요] 흡착된 물체가 있다면 물리 시뮬레이션 직후 구체 중심으로 포즈를 갱신해 줍니다.
    franka_ctrl.update_snapping_object()

    franka_ctrl.draw_ee_debug_sphere(viewer, radius=0.02, color=(1.0, 0.0, 0.0))

    franka_ctrl.draw_z_rotational_circle(
        viewer, 
        target_relative_xyz, 
        num_spheres=20, 
        sphere_radius=0.04, 
        color=(0.0, 0.5, 1.0)  # 스카이 블루 색상
    )

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
	
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)