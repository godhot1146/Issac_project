import os
import numpy as np
from isaacgym import gymapi, gymutil
from low_amr_controller_v2 import LowAMR

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

##### ==================== #####
# --- 로봇 인스턴스 스폰 ---
##### ==================== #####

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/low_move_rack/low_move_rack.urdf", move_rack_opts)
move_rack_handle = gym.create_actor(env, move_rack_asset, gymapi.Transform(p=gymapi.Vec3(0.0, -2.0, 0.05)), "move_rack_asset", -1, 0)

box_size = 0.5
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False

# 원하는 상자 무게 입력 (예: 5.0 kg)
target_box_mass = 10.0 

# 박스의 부피 계산 (V = 가로 * 세로 * 높이)
box_volume = box_size * box_size * box_size  # 0.125 m^3

# 밀도 공식: 밀도 = 질량 / 부피
# 5.0kg / 0.125m^3 = 40.0 kg/m^3
box_opts.density = target_box_mass / box_volume 

# 물리 안정성을 위한 선형/각속도 댐핑 추가 (공기 저항 효과)
box_opts.linear_damping = 0.2
box_opts.angular_damping = 0.2

# 에셋 생성 (지정한 밀도에 맞춰 PhysX가 질량 5kg과 그에 맞는 관성을 완벽히 자동 셋업합니다)
box_asset = gym.create_box(sim, box_size, box_size, box_size, box_opts)

b_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 3.0))
# b_handle = gym.create_actor(env, box_asset, b_pose, "cargo_box", -1, 0)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False

low_asset = gym.load_asset(sim, asset_root, "urdf/low/v2/low_amr_v2.urdf", low_opts)
low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.0)), "low_asset", -1, 1)

low_amr = LowAMR(gym, sim, env, low_handle)

waypoint_list = [
    (-1.0, 0.0),    # 1번 목적지
    (2.0, 1.0),    # 2번 목적지
    (-1.5, -1.0),   # 3번 목적지
    (0.0, 0.0)     # 복귀
]
current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 0.0          # 🎯 초기 목표 높이를 URDF 한계인 최고점(0.2m)으로 변경
LIFT_THRESHOLD = 0.005     # 🎯 가동 범위가 좁으므로 도달 공차를 5mm로 정밀화

# =========================================================================

# =========================================================================
# 🎯 리프트 회전(lift_down_revolute_joint) 기능 테스트용 변수
# =========================================================================
ROTATE_TEST_TARGETS = [0.0, np.pi / 2.0, np.pi, np.pi / 2.0, 0.0]  # 0° -> 90° -> 180° -> 90° -> 0° 순환
ROTATE_TEST_THRESHOLD = 0.02    # 도달 인정 각도 공차 (rad)
ROTATE_HOLD_FRAMES = 90         # 목표각 도달 후 다음 각도로 넘어가기 전 대기 프레임 (약 1.5초)
 
rotate_test_idx = 0
rotate_hold_start_frame = None

low_amr.set_state(0)
 
# =========================================================================

waypoint_list = [
    (0.0, -2.0, '-y', 'BACKWARD'),
    (0.0, -2.0, '-y', 'BACKWARD'),
    (0.0, -1.0, '-y', 'BACKWARD'), 
    (1.0, 1.0, '-y', 'BACKWARD')
]

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # low_amr.lift_and_locate_fsm_v3(
    #     frame_count=frame_count,
    #     waypoint_list=waypoint_list
    # )
	
    # =====================================================================
    # 🎯 [리프트 회전 왕복 테스트] lift_down_revolute_joint를 0° ↔ 180°로
    #    반복 왕복시켜, 회전 조인트 단독 동작(강성/댐핑/최대속도 튜닝)을
    #    검증하기 위한 테스트 루틴입니다.
    #    ROTATE_TEST_TARGETS 리스트를 순서대로 순환하며, 목표 각도에 도달하면
    #    ROTATE_HOLD_FRAMES(약 1.5초) 동안 대기한 뒤 다음 목표로 넘어갑니다.
    # =====================================================================
    current_rotate_target = ROTATE_TEST_TARGETS[rotate_test_idx]
    reached = low_amr.rotate_lift_to_angle(current_rotate_target, threshold=ROTATE_TEST_THRESHOLD)

    if reached:
        if rotate_hold_start_frame is None:
            rotate_hold_start_frame = frame_count
            print(f"[리프트 회전 테스트] 목표 각도 {np.degrees(current_rotate_target):.1f}° 도달 ➡️ {ROTATE_HOLD_FRAMES}프레임 대기")

        if (frame_count - rotate_hold_start_frame) >= ROTATE_HOLD_FRAMES:
            rotate_test_idx = (rotate_test_idx + 1) % len(ROTATE_TEST_TARGETS)
            rotate_hold_start_frame = None
            print(f"[리프트 회전 테스트] 다음 목표 각도로 전환 ➡️ {np.degrees(ROTATE_TEST_TARGETS[rotate_test_idx]):.1f}°")

    # 🎯 위에서 세팅한 리프트 회전 목표를 실제 액추에이터에 주입
    low_amr.apply_actuator_commands()
   
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)