import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController
from carter_amr import CarterAMR
from low_amr import LowProfileAMR
from a_star_test import AStarPlanner
from cardboard_box_manager import CardboardBoxManager

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
gym.set_light_parameters(sim, 0, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.0, 0.0, 0.0))

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

# --- 로봇 인스턴스 스폰 ---
desk_opts = gymapi.AssetOptions()
desk_opts.fix_base_link = True
desk_opts.density = 100.0
desk_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/desk.urdf", desk_opts)

# ==============================================================================
# 🛠️ [수정] 기구부(Caser, Taping) 전용 관절 활성화 옵션 추가
# ==============================================================================
machinery_opts = gymapi.AssetOptions()
machinery_opts.fix_base_link = True           # 베이스는 바닥 공간에 고정
machinery_opts.density = 100.0
machinery_opts.armature = 0.01                # 관절 모터 구동을 위한 미세 관성 보정
machinery_opts.linear_damping = 0.5
machinery_opts.angular_damping = 0.5
machinery_opts.disable_gravity = True         # 기구 자체 무게로 처지는 것 방지

fixed_caser_asset = gym.load_asset(sim, asset_root, "urdf/fixed_caser/v1/fixed_caser_v1.urdf", machinery_opts)
fixed_caser_handle = gym.create_actor(env, fixed_caser_asset, gymapi.Transform(p=gymapi.Vec3(-1.9, 0.0, 0.07)), "fixed_caser_asset", -1, 0)

taping_asset = gym.load_asset(sim, asset_root, "urdf/taping/v1/taping_v1.urdf", machinery_opts)
taping_handle = gym.create_actor(env, taping_asset, gymapi.Transform(p=gymapi.Vec3(-1.0, 0.0, 0.3)), "taping_asset", -1, 0)

# ==============================================================================
# 🛠️ [수정] 모든 구동부 관절 제어 설정을 taping_handle 기준으로 통합 초기화
# ==============================================================================
# 1. taping_handle 관절 프로퍼티 데이터 로드
tp_dof_props = gym.get_actor_dof_properties(env, taping_handle)
num_tp_dofs = len(tp_dof_props)

# 2. taping_handle 내부에서 Cutter, Front, Back DOF 인덱스를 각각 탐색
fc_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_cutter_joint", gymapi.DOMAIN_ACTOR)
tp_front_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_taping_front_joint", gymapi.DOMAIN_ACTOR)
tp_back_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_taping_back_joint", gymapi.DOMAIN_ACTOR)

# 디버깅 출력
print(f"📦 [Taping 기구부 관절 탐색 완료]")
print(f" - Cutter DOF Index: {fc_dof_idx}")
print(f" - Front Taping DOF Index: {tp_front_dof_idx}")
print(f" - Back Taping DOF Index: {tp_back_dof_idx}")
print(f" - 총 유효 DOF 개수: {num_tp_dofs}")

# 3. 발견된 모든 관절축을 위치 제어(POS_MODE) 고강성 모터 프로퍼티로 일괄 세팅
for i in range(num_tp_dofs):
    tp_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    tp_dof_props['stiffness'][i] = 15000.0  # Cutter 구동력 보강을 위해 상향 조정
    tp_dof_props['damping'][i] = 150.0

if num_tp_dofs > 0:
    gym.set_actor_dof_properties(env, taping_handle, tp_dof_props)

current_time = 0.0
# ==============================================================================

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

#box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(-3.0, 1.5, 1.5)), "box_asset", -1, 0)
box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4+0.8, 1.5)), "box_asset", -1, 0)

global_box_body_index = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_ENV)

box_manager = CardboardBoxManager(gym, sim, env, asset_root, pose=gymapi.Transform(p=gymapi.Vec3(-2.45, 0.4, 1.31), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))))

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = False
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v1/conveyor_v1.urdf", conveyor_opts)
conveyor_handle = gym.create_actor(env, conveyor_rack_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 0.0, 0.1)), "conveyor_asset", -1, 0)

# 🛠️ [수정] find_actor_rigid_body_index 함수로 통일하되, 마지막 도메인 인자만 다르게 지정합니다.
global_belt_index = gym.find_actor_rigid_body_index(env, conveyor_handle, "belt_link", gymapi.DOMAIN_ENV)
local_belt_index = gym.find_actor_rigid_body_index(env, conveyor_handle, "belt_link", gymapi.DOMAIN_ACTOR)

# 에셋 탐색 예외 처리 (이름을 못 찾으면 -1을 반환하므로 안전하게 0번으로 복구)
if global_belt_index == -1:
    print("경고: 글로벌 환경에서 'belt_link'를 찾지 못했습니다. 기본값 0을 적용합니다.")
    global_belt_index = 0
if local_belt_index == -1:
    print("경고: 액터 내부에서 'belt_link'를 찾지 못했습니다. 기본값 0을 적용합니다.")
    local_belt_index = 0

# 🛠️ [수정] anim 폴더 안의 128개 스크롤 텍스처를 리스트 컴프리헨션으로 자동 로드
anim_dir = os.path.join(asset_root, "urdf/conveyor/v1/anim")
texture_handles = [
    gym.create_texture_from_file(sim, os.path.join(anim_dir, f"belt_frame_{i:03d}.png"))
    for i in range(128)
]

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

vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120.urdf", vacuum_gripper_opts)
vacuum_gripper_asset_v2 = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120_box.urdf", vacuum_gripper_opts)

# 1. 스폰 위치 변수 지정
franka_pos = gymapi.Vec3(3.0, 2.0 + 0.4, 1.3)

# 2. Z축(0, 0, 1)을 기준으로 90도(라디안) 회전하는 쿼터니언 생성
#    np.radians(90.0) 대신 물리적으로 정확한 np.pi / 2.0 를 사용해도 좋습니다.
franka_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))

# 3. 위치와 회전값을 결합하여 Transform 생성
franka_transform = gymapi.Transform(p=franka_pos, r=franka_rot)

franka_pos_2 = gymapi.Vec3(-1.7, 0.6, 0.1)
franka_rot_2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
franka_transform_2 = gymapi.Transform(p=franka_pos_2, r=franka_rot_2)

# 4. 90도 회전된 상태로 Franka 로봇 스폰
franka_handle = gym.create_actor(env, vacuum_gripper_asset, franka_transform, "vacuum_gripper_asset", -1, 0)
franka_handle_2 = gym.create_actor(env, vacuum_gripper_asset_v2, franka_transform_2, "vacuum_gripper_asset", -1, 1)
desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(-2.7, 0.6, 0.0)), f"desk_asset", -1, 0)

pump_index = gym.find_actor_rigid_body_index(env, franka_handle, "cobot_pump", gymapi.DOMAIN_ACTOR)

desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4, 0.3)), f"desk_asset", -1, 0)
desk_handle2 = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4+0.8, 0.0)), f"desk_asset2", -1, 0)


if pump_index == -1:
    print("cobot_pump 링크를 찾을 수 없습니다. URDF 이름을 확인하세요.")
else:
    print(f"cobot_pump의 액터 내 인덱스: {pump_index}")

# --- 컨트롤러 연동 ---
franka_ctrl = FrankaController(gym, sim, env, franka_handle, pump_link_name="cobot_pump", scale=1.6)
franka_ctrl_v2 = FrankaController(gym, sim, env, franka_handle_2, pump_link_name="cobot_pump_base", scale=1.2)
franka_ctrl_v2.use_manual = True

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-1.64, -1.64, 1.70), gymapi.Vec3(-1.83, 1.06, 0.41))

# 감지할 키 이벤트 등록 (예: SPACE 키를 "start_process"라는 이름의 이벤트로 등록)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")
# 필요하다면 다른 키도 추가 가능 (예: R 키를 리셋용으로 등록)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_process")

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "select_prev_joint") # 1번 키
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "select_next_joint") # 2번 키
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "toggle_auto_rotation") # 3번 키

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "box_stage_0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "box_stage_1")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_7, "box_stage_2")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_8, "box_stage_3")

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_manual_mode") # 🛠️ 추가
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "decrease_joint_angle") # 🛠️ 이름 변경 (의도 매핑)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "increase_joint_angle") # 🛠️ 추가 (4번 키 각도 증가)

rotation_time_accumulator = 0.0

# 프로세스 제어를 위한 상태 플래그 변수 생성
is_active = False  # 스페이스바를 누르기 전까지는 로봇이 움직이지 않도록 보관

is_loaded = False
frame_count = 0

franka_ctrl.start_step = 0  
franka_ctrl.end_step = 1   
franka_ctrl.pause_duration = 5

robot_state = "MOVE_DOWN_1"    # 초기 상태: 첫 번째 하강
robot_state = "END"
step_dur = 50                 # 각 스테이트별 구동 시간

robot_state_v2 = "BOX_CATCH_START"

# target_relative_xyz = [0.6436, -0.0028, -0.0148] # 1배 스케일링
target_relative_xyz = [0.7706, -0.0008, 0.0065] # 1.2배 스케일링

robot_base_pos = np.array([-0.9, 0.0])
box_pos = np.array([-0.3, 0.2])

# 베이스에서 박스를 바라보는 상대 벡터 (dx, dy)
delta_pos = box_pos - robot_base_pos

# 💡 1번 관절이 회전해야 할 정확한 라디안 각도 계산 (atan2 활용)
target_box_angle = np.arctan2(delta_pos[1], delta_pos[0])

# 🛠️ [반반 업데이트 3] 애니메이션 속도 커스텀 설정 변수
animation_speed_frames = 5  # 몇 프레임 주기로 이미지를 바꿀지 지정 (낮을수록 고속 스크롤)

# --- [while 루프 진입 전 초기화 변수 추가] ---
L = 3.5                 # 컨베이어 벨트의 실제 물리 길이 (m) -> 본인 에셋에 맞게 수정!
P = 128                 # 텍스처 해상도 (128)
v = 0.5                # 목표 벨트 속도 (m/s)
dt = 1.0 / 60.0         # 1프레임당 시간

# 공식에 의해 계산된 프레임당 정밀 픽셀 이동량 (float)
pixels_per_frame = (v * dt * P) / L  

# 누적 픽셀 위치를 저장할 변수
accumulated_pixel_shift = 0.0

prev_box_z = 999.0      # 직전 프레임의 상자 Z 높이 저장용
is_landed = False       # 컨베이어 착지 여부 플래그

is_grabed = False

# fc_dof_idx = gym.find_actor_dof_index(env, fixed_caser_handle, "base_to_cutter_joint", gymapi.DOMAIN_ACTOR)
print(f"DEBUG: Cutter DOF Index = {fc_dof_idx}") # <-- 이 값이 -1인지 확인!

# ==============================================================================
# 0. 초기화 섹션 (클래스 생성부나 상태 초기화 섹션에 배치하세요)
# ==============================================================================
if not hasattr(franka_ctrl, 'is_holding'):
    # 처음 시작할 때 로봇이 물체를 안 들고 있다고 가정 (0도 위치에서 Pick 예정)
    franka_ctrl.is_holding = False 

while not gym.query_viewer_has_closed(viewer):
    # ==============================================================================
    # 🛠️ [수정] 키보드 이벤트 분기 처리 (1, 2, 3번 입력 인터셉트)
    # ==============================================================================
    for event in gym.query_viewer_action_events(viewer):
        if event.value > 0: # 키를 누르는 순간에만 작동 (Release 제외)
            # box_manager.handle_keyboard_event(event.action)

            # 🛠️ Franka 수동 조작 핸들러로 이벤트를 바로 패스시킵니다.
            # (내부에서 수동 모드 플래그 여부 및 조인트 변경/출력 연산을 모두 알아서 처리합니다.)
            franka_ctrl.handle_keyboard_event(event.action)
            franka_ctrl_v2.handle_keyboard_event(event.action) # 복수 로봇 제어 시 함께 연동 가능

            if event.action == "box_stage_0":
                box_manager.stage_0_unfolded_flat()
                print("상태 0: 박스를 완전히 펼쳤습니다.")
            elif event.action == "box_stage_1":
                box_manager.stage_1_fold_sides()
                print("상태 1: 사이드 사각 벽면 조립 완료.")
            elif event.action == "box_stage_2":
                box_manager.stage_2_close_bottom()
                print("상태 2: 하단 바닥 날개 밀봉 완료.")
            elif event.action == "box_stage_3":
                box_manager.stage_3_close_top()
                print("상태 3: 상단 날개 밀봉 완료 (박스 완성).")

            if event.action == "start_process":
                # 기존 카메라 보정 로그 기능 유지
                cam_matrix = gym.get_viewer_camera_transform(viewer, env)
                cam_pos = cam_matrix.p
                r = cam_matrix.r
                qx, qy, qz, qw = r.x, r.y, r.z, r.w
                forward_x = 2.0 * (qx * qz + qw * qy)
                forward_y = 2.0 * (qy * qz - qw * qx)
                forward_z = 1.0 - 2.0 * (qx * qx + qy * qy)
                target_dist = 3.0
                target_x = cam_pos.x + forward_x * target_dist
                target_y = cam_pos.y + forward_y * target_dist
                target_z = cam_pos.z + forward_z * target_dist
                print(f"\ngym.viewer_camera_look_at(viewer, env, gymapi.Vec3({cam_pos.x:.2f}, {cam_pos.y:.2f}, {cam_pos.z:.2f}), gymapi.Vec3({target_x:.2f}, {target_y:.2f}, {target_z:.2f}))")

    gym.clear_lines(viewer)
    
    # box_manager.draw_debug_visuals(viewer)

    # 2. 물리 엔진(simulate) 연산 전 관절 목표 각도 버퍼 최신화
    # box_manager.update_joints(dt=1.0/60.0)
    box_manager.update_joints()

    # 🛠️ 상사와 컨베이어 벨트의 실시간 위치 정보 획득
    box_transform = gym.get_rigid_transform(env, global_box_body_index)
    conveyor_transform = gym.get_rigid_transform(env, global_belt_index)
	
    current_box_z = box_transform.p.z
    z_diff = abs(current_box_z - prev_box_z)

	# 🛠️ 컨베이어 벨트 중심 기준 상자와의 Y축 상대 거리 계산 (relative_y = 상자_y - 벨트_y)
    relative_y = box_transform.p.y - conveyor_transform.p.y

    # 상자가 떨어지는 중이 아니고(z_diff가 매우 작음) 
    # 컨베이어 표면 높이(예: 대략 0.1m 이상 바닥 위)에 도달했을 때 착지로 판정
    # 0.0001 임계값은 물리 시뮬레이션 오차를 고려한 수치입니다.
    if not is_landed and z_diff < 0.0001 and current_box_z > 0.05:
        # 최초 안착 순간 프레임 카운트 등 조건이 누적되면 완벽히 착지한 것으로 고정
        is_landed = True
        print(f"상자 안착 감지! 현재 Z 높이: {current_box_z:.4f}m -> 벨트 가동 시작")

    # 🛠️ 착지가 완료된 상태에서만 상자 속도 주입 및 텍스처 스크롤 가동
    if is_landed and (-1.8 <= relative_y <= 1.8) and not is_grabed:
        # 1. 기존 속도 주입
        gym.set_rigid_linear_velocity(env, global_box_body_index, gymapi.Vec3(0.0, -v-0.2, 0.0))
        gym.set_rigid_angular_velocity(env, global_box_body_index, gymapi.Vec3(0.0, 0.0, 0.0))

        # 💡 [핵심 수정] 단일 transform을 변경하는 대신, 액터 전체의 rigid_body_states를 안전하게 가져옵니다.
        box_states = gym.get_actor_rigid_body_states(env, box_handle, gymapi.STATE_ALL)
        
        # 루트(0번 인덱스)의 회전만 순수 정자세(Quat 0,0,0,1)로 리셋합니다.
        box_states['pose'][0]['r']['x'] = 0.0
        box_states['pose'][0]['r']['y'] = 0.0
        box_states['pose'][0]['r']['z'] = 0.0
        box_states['pose'][0]['r']['w'] = 1.0

        # Articulation 규격을 만족하기 위해 모든 자식 링크들의 속도 상태를 안전하게 순회하며 배열을 채웁니다.
        num_bodies = len(box_states)
        for i in range(num_bodies):
            # 컨베이어 위에서 미끄러지는 선속도 보정 (필요 시 유지)
            if i == 0:
                box_states['vel'][i]['linear']['y'] = -v - 0.2
            else:
                box_states['vel'][i]['linear']['x'] = 0.0
                box_states['vel'][i]['linear']['y'] = 0.0
                box_states['vel'][i]['linear']['z'] = 0.0
                
            box_states['vel'][i]['angular']['x'] = 0.0
            box_states['vel'][i]['angular']['y'] = 0.0
            box_states['vel'][i]['angular']['z'] = 0.0

        # 온전한 배열 구조를 통째로 주입하여 경고를 완벽히 제거합니다.
        gym.set_actor_rigid_body_states(env, box_handle, box_states, gymapi.STATE_ALL)

        # 매 프레임마다 계산된 소수점 픽셀 수만큼 누적 합산하여 텍스처 변경
        accumulated_pixel_shift += pixels_per_frame
        current_tex_idx = int(accumulated_pixel_shift) % len(texture_handles)
    elif relative_y < -1.8:
        accumulated_pixel_shift += pixels_per_frame
        current_tex_idx = int(accumulated_pixel_shift) % len(texture_handles)
        
    else:
        # 아직 낙하 중일 때는 상자 속도를 물리 엔진 자유 낙하에 맡기고, 텍스처는 0번에 고정(정지)
        current_tex_idx = 0

    # 현재 Z 좌표를 다음 프레임 비교를 위해 저장
    prev_box_z = current_box_z
    
    # 3. 텍스처 주입
    gym.set_rigid_body_texture(
        env, 
        conveyor_handle, 
        local_belt_index,  
        gymapi.MESH_VISUAL, 
        texture_handles[current_tex_idx]
    )

    # ==============================================================================
    # 🛠️ [수정] 하나의 단일 taping_handle 내부 배열로 통합 제어 신호 주입
    # ==============================================================================
    current_time += dt  # 시간 누적

    num_dofs = gym.get_actor_dof_count(env, taping_handle)
    if num_dofs > 0:
        # 단일 액터(taping_handle) 전체 관절 개수만큼 버퍼 할당
        taping_targets = np.zeros(num_dofs, dtype=np.float32)
        
        # 1. 커터(Cutter) 왕복 사인파 제어 주입
        if fc_dof_idx != -1 and fc_dof_idx < num_dofs:
            taping_targets[fc_dof_idx] = 1.0 * np.sin(2.0 * current_time)
            
        # 2. 테이핑 앞(Front) 왕복 구동 주입
        if tp_front_dof_idx != -1 and tp_front_dof_idx < num_dofs:
            taping_targets[tp_front_dof_idx] = 0.8 * np.sin(3.0 * current_time)
            
        # 3. 테이핑 뒤(Back) 왕복 구동 주입 
        if tp_back_dof_idx != -1 and tp_back_dof_idx < num_dofs:
            taping_targets[tp_back_dof_idx] = 0.8 * np.cos(3.0 * current_time)
            
        # 단 한번의 API 호출로 모든 관절 구동축 동기화 명령 하사
        gym.set_actor_dof_position_targets(env, taping_handle, taping_targets)
    # ==============================================================================

    # --------------------------------------------------------------------------
    # 🛠️ 수동 모드일 경우 계산된 유저 타겟 버퍼를 하드웨어 타겟에 주입합니다.
    # (자동 모드일 때는 하위의 FSM 내부 if문 조건에 의해 주입되므로 충돌하지 않습니다.)
    franka_ctrl_v2.apply_manual_targets()
    # --------------------------------------------------------------------------

    # 예시: 로봇 상태 분기 처리 구조 내에 결합
    if robot_state_v2 == "BOX_CATCH_START":
        # 지정한 구동 속도(step_dur)에 맞춰 상자 파지 자세 시퀀스 수행
        if franka_ctrl_v2.box_catch_sequence(step_duration=step_dur):
            print("[상태 전환] 상자 접근 자세 완수 ➡️ 물체 흡착(Attach) 및 상승 단계 진입")
            
            # 시퀀스가 끝나는 타이밍에 맞춰 물리적으로 흡착 인터록을 겁니다.
            # franka_ctrl_v2.attach(box_handle, distance_threshold=0.15)
            
            # 다음 FSM 상태로 전환
            robot_state_v2 = "BOX_CATCH_ROTATE"

    elif robot_state_v2 == "BOX_CATCH_ROTATE":
        # 지정한 구동 속도(step_dur)에 맞춰 상자 파지 자세 시퀀스 수행
        rot_frame = franka_ctrl_v2.rotate_link1(-1.571)
        if rot_frame >= (step_dur + franka_ctrl_v2.pause_duration):
            robot_state_v2 = "BOX_CLOSE_ROTATE"
            print("[상태 전환] 90도 회전 완료")

    elif robot_state_v2 == "BOX_CLOSE_ROTATE":
        # 지정한 구동 속도(step_dur)에 맞춰 상자 파지 자세 시퀀스 수행
        if franka_ctrl_v2.box_rotate_sequence(step_duration=step_dur):
            print("[상태 전환] 상자 접근 회전 완수")
            franka_ctrl_v2.attach(box_manager.handle, distance_threshold=0.25)
            robot_state_v2 = "BOX_CATCH_ROTATE2"

    elif robot_state_v2 == "BOX_CATCH_ROTATE2":
        # 지정한 구동 속도(step_dur)에 맞춰 상자 파지 자세 시퀀스 수행
        rot_frame = franka_ctrl_v2.rotate_link1(0.0)
        if rot_frame >= (step_dur + franka_ctrl_v2.pause_duration):
            robot_state_v2 = "END"
            print("[상태 전환] 90도 회전 완료")

    # ==============================================================================
    # FSM 메인 시퀀스 루프
    # ==============================================================================
    
    # 💡 [중요] 흡착된 물체가 있다면 물리 시뮬레이션 직후 구체 중심으로 포즈를 갱신해 줍니다.
    franka_ctrl.update_snapping_object()

    # 1. [0도 위치] 하강 시퀀스 (Pick 또는 Place 토글)
    if robot_state == "MOVE_DOWN_1":
        if franka_ctrl.move_down_sequence(step_duration=step_dur):
            rel_xyz = franka_ctrl.get_sphere_position_relative_to_base()
            if rel_xyz is not None:
                print(f"🎯 0도 하강 완료! 베이스 기준 상대 좌표 -> X: {rel_xyz[0]:.4f}, Y: {rel_xyz[1]:.4f}, Z: {rel_xyz[2]:.4f}")
            franka_ctrl.attach(box_handle, distance_threshold=0.15)
            is_grabed = True
            robot_state = "MOVE_UP_1"
            print("[상태 전환] 0도 위치 하강 작업 완수 ➡️ STATE_MOVE_UP_1 단계 진입")
            
    # 2. [0도 위치] 상승 시퀀스
    elif robot_state == "MOVE_UP_1":
        if franka_ctrl.move_up_sequence(step_duration=step_dur):
            robot_state = "ROTATE_90"
            print("[상태 전환] 0도 위치 상승 완료 ➡️ 90도 회전 시작")

    # 3. [회전] 1번 관절을 90도(약 1.571 라디안)로 회전 및 대기
    elif robot_state == "ROTATE_90":
        rot_frame = franka_ctrl.rotate_link1(1.571*2)
        if rot_frame >= (step_dur + franka_ctrl.pause_duration):
            robot_state = "MOVE_DOWN_2"
            print("[상태 전환] 90도 회전 완료 ➡️ 90도 위치 하강 시작")

    # 4. [90도 위치] 하강 시퀀스 (Place 또는 Pick 토글)
    elif robot_state == "MOVE_DOWN_2":
        if franka_ctrl.move_down_sequence(step_duration=step_dur):
            franka_ctrl.detach()
            robot_state = "MOVE_UP_2"
            print("[상태 전환] 90도 위치 하강 작업 완수 ➡️ STATE_MOVE_UP_2 단계 진입")

    # 5. [90도 위치] 상승 시퀀스
    elif robot_state == "MOVE_UP_2":
        if franka_ctrl.move_up_sequence(step_duration=step_dur):
            robot_state = "ROTATE_RETURN"
            is_grabed = False

            print("[상태 전환] 90도 위치 상승 완료 ➡️ 0도 원위치 복귀 회전 시작")

    # 6. [원위치 복귀 회전] 1번 관절을 다시 0도로 돌려놓기
    elif robot_state == "ROTATE_RETURN":
        rot_frame = franka_ctrl.rotate_link1(0.0)
        if rot_frame >= (step_dur + franka_ctrl.pause_duration):
            robot_state = "END"

    franka_ctrl.draw_ee_debug_sphere(viewer, radius=0.02, color=(1.0, 0.0, 0.0))
    franka_ctrl_v2.update_snapping_object()
    franka_ctrl_v2.draw_ee_debug_sphere(viewer, radius=0.02, color=(1.0, 0.0, 0.0))
    # franka_ctrl.draw_z_rotational_circle(
    #     viewer, 
    #     target_relative_xyz, 
    #     num_spheres=20, 
    #     sphere_radius=0.04, 
    #     color=(0.0, 0.5, 1.0)  # 스카이 블루 색상
    # )

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
	
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)