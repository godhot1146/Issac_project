import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController
from cardboard_box_manager import CardboardBoxManager
from low_amr_controller_v1 import LowAMR
from forklift_amr_controller_v1 import ForkliftAMR
from indy7_controller_v1 import *
from scipy.spatial.transform import Rotation as R

def draw_actor_origin_debug_sphere(viewer, actor_handle, radius=0.05, color=(1.0, 0.0, 0.0)):
    """
    지정된 액터 핸들의 리지드 바디 상태를 추적하여 실시간 원점에 디버그 구체를 그립니다.
    
    :param viewer: Isaac Gym viewer 인스턴스
    :param actor_handle: 구체를 표시할 대상 액터 핸들 (예: dummy_handle1)
    :param radius: 구체의 기본 반지름
    :param color: (R, G, B) 색상 튜플 (기본값: 파란색)
    """
    # 1. 액터의 순수한 리지드 바디 상태 배열을 월드 좌표계 기준으로 획득
    body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    
    if body_states is None or len(body_states) == 0:
        return

    # 2. 액터의 기준 원점(일반적으로 루트 리지드 바디인 0번 인덱스) 추출
    root_link_idx = 0
    pose = body_states['pose'][root_link_idx]
    
    # 실시간 월드 좌표 Vec3 변환
    origin = gymapi.Vec3(pose['p']['x'], pose['p']['y'], pose['p']['z'])
    
    # 3. gymutil 와이어프레임 구체 객체 생성
    sphere_geom = gymutil.WireframeSphereGeometry(
        radius * 1.4, 
        10, 
        10, 
        gymapi.Transform(p=origin), 
        color
    )
    
    # 4. 뷰어 화면에 최종 드로우
    gymutil.draw_lines(sphere_geom, gym, viewer, env, gymapi.Transform())

def actor_origin(actor_handle):
    # 1. 액터의 순수한 리지드 바디 상태 배열을 월드 좌표계 기준으로 획득
    body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    
    if body_states is None or len(body_states) == 0:
        return

    # 2. 액터의 기준 원점(일반적으로 루트 리지드 바디인 0번 인덱스) 추출
    root_link_idx = 0
    pose = body_states['pose'][root_link_idx]
    
    # 실시간 월드 좌표 Vec3 변환
    origin = gymapi.Vec3(pose['p']['x'], pose['p']['y'], pose['p']['z'])

    return origin

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
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

room_size = 9.1          
wall_height = 10.0        
wall_thickness = 0.2     

floor_asset = gym.create_box(sim, room_size, room_size, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_size, wall_height, env_opts)
wall_y = gym.create_box(sim, room_size, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height, env_opts) 

# =================================================================
# 4. 25평 오픈 팩토리 환경 생성 및 외벽/기둥 배치 (Actors)
# =================================================================
env = gym.create_env(sim, gymapi.Vec3(-room_size/2, -room_size/2, 0), gymapi.Vec3(room_size/2, room_size/2, room_size), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

half_r = room_size / 2
w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_r, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_r, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_r, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_r, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
	gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

p_offset = half_r - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset, p_offset]:
	for y in [-p_offset, p_offset]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

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

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

box_pose1 = gymapi.Transform(p=gymapi.Vec3(-0.2, -0.45, 0.95), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
box_manager1 = CardboardBoxManager(gym, sim, env, asset_root, pose=box_pose1, fix=False)

box_pose2 = gymapi.Transform(p=gymapi.Vec3(-0.8, -0.45, 0.95), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
box_manager2 = CardboardBoxManager(gym, sim, env, asset_root, pose=box_pose2, fix=False)

box_pose3 = gymapi.Transform(p=gymapi.Vec3(-1.4, -0.45, 0.95), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
box_manager3 = CardboardBoxManager(gym, sim, env, asset_root, pose=box_pose3, fix=False)

box_pose4 = gymapi.Transform(p=gymapi.Vec3(-2.0, -0.45, 0.95), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
box_manager4 = CardboardBoxManager(gym, sim, env, asset_root, pose=box_pose4, fix=False)

box_pose5 = gymapi.Transform(p=gymapi.Vec3(-2.6, -0.45, 0.95), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
box_manager5 = CardboardBoxManager(gym, sim, env, asset_root, pose=box_pose5, fix=False)

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = True
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v2/conveyor_v2.urdf", conveyor_opts)

conveyor_pos = gymapi.Vec3(-1.6, -0.6, 0.04)
conveyor_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
conveyor_transform = gymapi.Transform(p=conveyor_pos, r=conveyor_rot)

conveyor_handle = gym.create_actor(env, conveyor_rack_asset, conveyor_transform, "conveyor_asset", -1, 0)

global_belt_index = gym.find_actor_rigid_body_index(env, conveyor_handle, "belt_link", gymapi.DOMAIN_ENV)
local_belt_index = gym.find_actor_rigid_body_index(env, conveyor_handle, "belt_link", gymapi.DOMAIN_ACTOR)

if global_belt_index == -1: global_belt_index = 0
if local_belt_index == -1: local_belt_index = 0

anim_dir = os.path.join(asset_root, "urdf/conveyor/v1/anim")
texture_handles = [
    gym.create_texture_from_file(sim, os.path.join(anim_dir, f"belt_frame_{i:03d}.png"))
    for i in range(128)
]

# ==============================================================================
# 🆕 [이식] 컨베이어 벨트 이동/텍스처 스크롤 설정 (구 main 스크립트에서 포팅)
#     - box_manager1~5 각각의 상태를 독립적으로 추적
#     - 컨베이어가 -90° 회전되어 있으므로, 판정/속도 주입은 컨베이어 "로컬 좌표계" 기준으로 수행
# ==============================================================================
L = 3.5                 # 컨베이어 벨트의 실제 물리 길이(m) — v2 에셋 실측값으로 조정 필요
P = 128                 # 텍스처 프레임 수
v = 0.5                 # 벨트 목표 속도 (m/s)
_conveyor_dt = 1.0 / 60.0
pixels_per_frame = -(v * _conveyor_dt * P) / L
accumulated_pixel_shift = 0.0

conveyor_running = False   # 🆕 컨베이어 가동 여부 (키로 토글)

CONVEYOR_MOVE_DISTANCE = 0.6     # 목표 이동 거리(m)
conveyor_moved_distance = 0.0
conveyor_start_box5_local_y = None   # 이동 시작 시점의 5번 박스 위치(컨베이어 로컬 기준)

box_managers = [box_manager1, box_manager2, box_manager3, box_manager4, box_manager5]

# 각 박스의 "루트" 리지드바디(0번, 박스 바닥/베이스) 글로벌 인덱스 확보
box_body_indices = [
    gym.get_actor_rigid_body_index(env, bm.handle, 0, gymapi.DOMAIN_ENV)
    for bm in box_managers
]

# 박스별 컨베이어 상태 추적 (착지 여부, 직전 z값)
box_conveyor_state = [
    {"prev_z": 999.0, "is_landed": False}
    for _ in box_managers
]

def _world_to_local(origin_pos, origin_rot, world_pos):
    """world_pos를 origin(위치/회전) 기준 로컬 좌표로 변환"""
    delta = gymapi.Vec3(world_pos.x - origin_pos.x,
                         world_pos.y - origin_pos.y,
                         world_pos.z - origin_pos.z)
    return origin_rot.inverse().rotate(delta)

def _flatten_keep_yaw(rot: gymapi.Quat) -> gymapi.Quat:
    """roll/pitch만 0으로 눌러 수평 정렬하고, yaw(z축 회전)는 현재 값 그대로 유지."""
    q = R.from_quat([rot.x, rot.y, rot.z, rot.w])
    _, _, yaw = q.as_euler('xyz', degrees=False)
    flat = R.from_euler('z', yaw)
    fx, fy, fz, fw = flat.as_quat()
    return gymapi.Quat(fx, fy, fz, fw)

# indy7_opts = gymapi.AssetOptions()
# indy7_opts.fix_base_link = True
# indy7_opts.density = 100.0
# indy7_asset = gym.load_asset(sim, asset_root, "urdf/indy_description/urdf_files/indy7_v3_eye _vacuum.urdf", indy7_opts)
# indy7_handle = gym.create_actor(env, indy7_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.5)), "indy7_asset", -1, 0)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v1/low_amr_v1.urdf", low_opts)
#low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 1.0, 0.2)), "low_asset", -1, 1)
#low_amr = LowAMR(gym, sim, env, low_handle)

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/low_move_rack/low_move_rack.urdf", move_rack_opts)
# move_rack_handle = gym.create_actor(env, move_rack_asset, gymapi.Transform(p=gymapi.Vec3(-2.0, 1.0, 0.05)), "move_rack_asset", -1, 0)

pallet_asset = gym.load_asset(sim, asset_root, "urdf/pallet/v2/pallet_v2.urdf", move_rack_opts)
pallet_handle = gym.create_actor(env, pallet_asset, gymapi.Transform(p=gymapi.Vec3(0.75, -0.1, 0.1)), "pallet_asset", -1, 0)

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False

forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)
# forklift_handle = gym.create_actor(env, forklift_asset, gymapi.Transform(p=gymapi.Vec3(2.0, 1.0, 0.3)), "forklift_asset", -1, 1)

# forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

# --- 컨트롤러 연동 및 스케일 바인딩 ---

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-1.64, -1.64, 1.70), gymapi.Vec3(-1.83, 1.06, 0.41))

indy7_transform = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.5), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
indy7_arm = IndyArmController(
        gym, sim, env, viewer,
        asset_root=asset_root,
        urdf_path="urdf/indy_description/urdf_files/indy7_v3_eye _vacuum.urdf",
        spawn_transform=indy7_transform,
    )
gym.prepare_sim(sim)
indy7_arm.setup_tensors()

# 🆕 gym.prepare_sim(sim) 이후, indy7_arm.setup_tensors() 다음에 위치시켜야 함
# (DOMAIN_SIM 인덱스는 prepare_sim 이후에 조회하는 것이 안전)

box_actor_sim_indices = [
    gym.get_actor_index(env, bm.handle, gymapi.DOMAIN_SIM) for bm in box_managers
]
box_rb_sim_indices = [
    gym.get_actor_rigid_body_index(env, bm.handle, 0, gymapi.DOMAIN_SIM) for bm in box_managers
]
belt_rb_sim_index = gym.find_actor_rigid_body_index(env, conveyor_handle, "belt_link", gymapi.DOMAIN_SIM)

indy7_arm.register_joint_pose("pick_start", [-2.9758, 0.4319, -1.2063, 0.0, -2.3863, -2.9307])
indy7_arm.register_joint_pose("pick_approach",     [-1.5774, -0.6191, -0.7429, 0.0121, -1.8039, -3.1370])
indy7_arm.register_joint_pose("pick_move", [0.0103, -0.6802, -0.4950, 0.0130, -1.9992, -3.1344])
indy7_arm.register_joint_pose("place_1", [0.8955, -0.3810, -2.1105, 0.0565, -0.6614, -2.2981])

pick_sequence = [
    ("pick_start", 0.5),   # 접근 (대기 없음, 도착 즉시 다음으로)
    ("pick_approach",     0.5),   # 하강 후 0.3초 대기 (흡착 안정화 시간)
]

indy7_arm.register_attachable(box_manager1.handle)
indy7_arm.register_attachable(box_manager2.handle)
# ... 필요한 만큼 등록

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "toggle_attach")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_conveyor")

# ===============================
##### low amr 설정 #####
# ===============================

shelf_target_pos = (-2.0, 1.0) # (-2.0, 1.0, 0.05)
delivery_pos     = (0.0, 1.0) # 0.0, 1.0

waypoint_list = [
    (0.0, 1.0, '+x', 'BACKWARD'),
    (-2.0, 1.0, '+x', 'BACKWARD'),
    (-1.0, 1.0, '+x', 'FORWARD'), 
    (0.0, 1.0, '+x', 'FORWARD')
]

manual_angular_cmd = 0.0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 0.2          # 초기 목표 높이를 URDF 한계인 최고점(0.2m)으로 변경
LIFT_TARGET_POS = 0.16  # 물리 상승 임계 상한선 타겟값
LIFT_THRESHOLD = 0.005     # 가동 범위가 좁으므로 도달 공차를 5mm로 정밀화

# ===============================
# 키보드 입력 구독 등록
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "select_prev_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "select_next_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "toggle_auto_rotation")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "box_stage_0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "box_stage_1")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_7, "box_stage_2")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_8, "box_stage_3")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_manual_mode") 
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "toggle_box_manual_mode") 
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "decrease_joint_angle") 
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "increase_joint_angle") 

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "box_move_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "box_move_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "box_move_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "box_move_right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "box_move_up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "box_move_down")

key_states = {
    "box_move_forward": False, "box_move_backward": False,
    "box_move_left": False, "box_move_right": False,
    "box_move_up": False, "box_move_down": False
}
m_key_released = True

# 🤖 [2호기] 초기화
robot_state_v2 = "CATCH_START"
step_dur = 1    
target_relative_xyz = [0.7706, -0.0008, 0.0065] # 1.2배 스케일링

sequencer = JointPoseSequencer(indy7_arm)
dt=1.0/60.0
state = "MOVE_TO_DOCK"
state_entered = False

# --- 안정화 체크용 변수 ---
STABLE_THRESHOLD = 0.0005   # 5mm 미만이면 "정지"로 간주 (필요시 조정)
STABLE_FRAMES_REQUIRED = 10 # 연속 10프레임 이상 안 움직이면 확정
stable_counter = 0
prev_ee_pos = None

def check_ee_stable(arm, prev_pos, threshold=0.0005):
    """
    현재 ee 위치를 읽어와 prev_pos와 비교.
    반환: (현재위치, 변화량이 threshold 미만인지 bool)
    """
    arm.gym.refresh_rigid_body_state_tensor(arm.sim)
    cur_pos = arm.rb_states[arm.ee_index_global, 0:3].clone()
    if prev_pos is None:
        return cur_pos, False
    delta = (cur_pos - prev_pos).norm().item()
    return cur_pos, (delta < threshold)

# ==============================================================================
# 🚀 메인 시뮬레이션 루프 진입
# ==============================================================================
frame_count = 0
while not gym.query_viewer_has_closed(viewer):
    
    # ① 키보드 입력 핸들링 및 다이렉트 호출 분기
    events = gym.query_viewer_action_events(viewer)

    for event in events:
        is_pressed = (event.value > 0)
        if event.action in key_states:
            key_states[event.action] = is_pressed
        if event.value > 0: 
            if event.action == "start_process":
                is_active = True
            elif event.action == "reset_process":
                is_active = False
        
        if event.value > 0 and event.action == "toggle_attach":
            if indy7_arm.is_attached:
                pass
                indy7_arm.detach()
            else:
                pass
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("attached_sim_index:", indy7_arm.attached_sim_index)
                print("rb_states shape:", indy7_arm.rb_states.shape)

        if event.value > 0 and event.action == "toggle_conveyor":
            if not conveyor_running:
                conveyor_running = True
                conveyor_start_box5_local_y = None   # 🔑 다음 프레임에 현재 위치로 새로 기록
                print(f"[Conveyor] ▶ 5번 박스 기준 {CONVEYOR_MOVE_DISTANCE:.2f}m 이동 시작")
            else:
                print("[Conveyor] 이미 이동 중입니다.")

    gym.clear_lines(viewer)

    # ==============================================================================
    # 🆕 [이식] 컨베이어 벨트 박스 이동 + 텍스처 스크롤
    # ==============================================================================
    indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
    indy7_arm.gym.refresh_actor_root_state_tensor(indy7_arm.sim)

    belt_pos = indy7_arm.rb_states[belt_rb_sim_index, 0:3]
    belt_rot = indy7_arm.rb_states[belt_rb_sim_index, 3:7]
    conveyor_r = gymapi.Quat(belt_rot[0].item(), belt_rot[1].item(), belt_rot[2].item(), belt_rot[3].item())
    conveyor_p = gymapi.Vec3(belt_pos[0].item(), belt_pos[1].item(), belt_pos[2].item())

    if conveyor_running:
        box5_rb_idx = box_rb_sim_indices[4]
        box5_pos = indy7_arm.rb_states[box5_rb_idx, 0:3]
        box5_world = gymapi.Vec3(box5_pos[0].item(), box5_pos[1].item(), box5_pos[2].item())
        box5_local = _world_to_local(conveyor_p, conveyor_r, box5_world)

        if conveyor_start_box5_local_y is None:
            conveyor_start_box5_local_y = box5_local.y

        conveyor_moved_distance = abs(box5_local.y - conveyor_start_box5_local_y)
        if conveyor_moved_distance >= CONVEYOR_MOVE_DISTANCE:
            conveyor_running = False
            print(f"[Conveyor] ⏸ {conveyor_moved_distance:.3f}m 이동 완료 → 자동 정지")

    any_box_moving = False
    updated_actor_indices = []

    for bm, rb_idx, actor_idx, cstate in zip(box_managers, box_rb_sim_indices, box_actor_sim_indices, box_conveyor_state):
        if rb_idx == -1 or actor_idx == -1:
            continue

        box_pos_t = indy7_arm.rb_states[rb_idx, 0:3]
        current_box_z = box_pos_t[2].item()
        z_diff = abs(current_box_z - cstate["prev_z"])

        box_world = gymapi.Vec3(box_pos_t[0].item(), box_pos_t[1].item(), current_box_z)
        local_pos = _world_to_local(conveyor_p, conveyor_r, box_world)
        relative_forward = local_pos.y

        is_grabbed = indy7_arm.is_attached and indy7_arm.attached_handle == bm.handle

        if not cstate["is_landed"] and z_diff < 0.0001 and current_box_z > 0.05:
            cstate["is_landed"] = True
            print(f"[Conveyor] 박스 안착 감지 (z={current_box_z:.4f}m)")

        new_state = indy7_arm.root_states[actor_idx].clone()

        if conveyor_running and cstate["is_landed"] and (-1.8 <= relative_forward <= 1.8) and not is_grabbed:
            belt_local_vel = gymapi.Vec3(0.0, +(v + 0.2), 0.0)
            world_vel = conveyor_r.rotate(belt_local_vel)

            cur_rot_t = indy7_arm.rb_states[rb_idx, 3:7]
            cur_q = gymapi.Quat(cur_rot_t[0].item(), cur_rot_t[1].item(), cur_rot_t[2].item(), cur_rot_t[3].item())
            flat_q = _flatten_keep_yaw(cur_q)

            new_state[3], new_state[4], new_state[5], new_state[6] = flat_q.x, flat_q.y, flat_q.z, flat_q.w
            new_state[7], new_state[8], new_state[9] = world_vel.x, world_vel.y, world_vel.z
            new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
            any_box_moving = True

        elif cstate["is_landed"] and not conveyor_running and not is_grabbed:
            new_state[7], new_state[8], new_state[9] = 0.0, 0.0, 0.0
            new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
        else:
            cstate["prev_z"] = current_box_z
            continue  # 변경 없음 → 굳이 텐서 인덱싱에 포함 안 함

        indy7_arm.root_states[actor_idx] = new_state
        updated_actor_indices.append(actor_idx)
        cstate["prev_z"] = current_box_z

    if updated_actor_indices:
        actor_indices_t = torch.tensor(updated_actor_indices, dtype=torch.int32, device=indy7_arm.root_states.device)
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(indy7_arm.root_states),
            gymtorch.unwrap_tensor(actor_indices_t),
            len(updated_actor_indices)
        )

    # 벨트 텍스처는 가동 중일 때만 스크롤, 정지 시 마지막 프레임에서 고정
    if conveyor_running and any_box_moving:
        accumulated_pixel_shift += pixels_per_frame
    current_tex_idx = int(accumulated_pixel_shift) % len(texture_handles)

    gym.set_rigid_body_texture(
        env, conveyor_handle, local_belt_index, gymapi.MESH_VISUAL, texture_handles[current_tex_idx]
    )

    if state == "MOVE_TO_DOCK":
        if not state_entered:
            print("[PickPlaceTask] pick_start 자세로 이동 시작")
            indy7_arm.move_to_joint_pose("pick_start")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PickPlaceTask] 완료")
                state = "PICK_SEQUENCE"
                state_entered = False

    elif state == "PICK_SEQUENCE":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_approach")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PickPlaceTask] 완료")
                state = "PICK_UP"
                state_entered = False

    elif state == "PICK_UP":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += 0.05   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PICK_MOVE"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

    elif state == "PICK_MOVE":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_move")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PICK_MOVE] 완료")
                state = "PLACE_1_1"
                state_entered = False

    elif state == "PLACE_1_1":
        if not state_entered:
            indy7_arm.motion_gen.angular_speed=0.2
            indy7_arm.move_to_joint_pose("place_1")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            cur_q = indy7_arm.dof_states_full[indy7_arm.dof_indices, 0]
            target_q = indy7_arm.joint_motion_gen.target_q
            print("오차(deg):", np.degrees((cur_q - target_q).cpu().numpy()))

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                indy7_arm.motion_gen.angular_speed=1.0
                print("[PLACE_1] 완료")
                state = "PLACE_1_2"
                state_entered = False

    elif state == "PLACE_1_2":
        if not state_entered:
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] -= 0.27   

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PLACE_1_2] 완료")
                indy7_arm.detach()
                state = "PICK_MOVE2"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

                
    elif state == "PICK_MOVE2":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_move")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PICK_MOVE] 완료")
                state = "PICK_SEQUENCE2"
                state_entered = False

    elif state == "PICK_SEQUENCE2":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_approach")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PickPlaceTask] 완료")
                state = "PICK_UP2"
                state_entered = False

    elif state == "PICK_UP2":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += 0.05   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP2] 완료 → 컨베이어 이동 시작")
                indy7_arm.motion_gen.linear_speed = 5.0

                # 🆕 컨베이어 이동 시작
                conveyor_running = True
                conveyor_start_box5_local_y = None   # 새 구간 기준점 리셋

                state = "WAIT_CONVEYOR2"
                state_entered = False

    elif state == "WAIT_CONVEYOR2":
        if not state_entered:
            print("[WAIT_CONVEYOR2] 컨베이어 이동 대기 중...")
            state_entered = True

        # 🔑 conveyor_running은 메인 루프 최상단 컨베이어 블록에서
        #     conveyor_moved_distance >= CONVEYOR_MOVE_DISTANCE가 되는 순간 자동으로 False가 됨
        if not conveyor_running:
            print("[WAIT_CONVEYOR2] 컨베이어 이동 완료 → PICK_SEQUENCE3로 전환")
            state = "PICK_SEQUENCE3"
            state_entered = False

    elif state == "PICK_SEQUENCE3":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_approach")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PickPlaceTask] 완료")
                state = "PICK_UP3"
                state_entered = False

    elif state == "PICK_UP3":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += 0.05   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PICK_MOVE3"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

    elif state == "PICK_MOVE3":
        if not state_entered:
            indy7_arm.move_to_joint_pose("pick_move")
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        if not indy7_arm.joint_motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0

            if stable_counter >= STABLE_FRAMES_REQUIRED:
                indy7_arm.attach_nearest(distance_threshold=0.3)
                print("[PICK_MOVE] 완료")
                state = "PLACE_2_1"
                state_entered = False

    elif state == "PLACE_2_1":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += -0.23   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PLACE_2_2"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

    elif state == "PLACE_2_2":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[1] += 0.36   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PLACE_2_3"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

    elif state == "PLACE_2_3":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[0] += 0.09   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PLACE_2_4"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0

    elif state == "PLACE_2_4":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += -0.389   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "PLACE_2_5"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0
                indy7_arm.detach()

    elif state == "PLACE_2_5":
        if not state_entered:
            # 현재 end-effector의 실제 위치/자세를 읽어와 그 자리에서 z만 10cm 올린 목표를 만든다
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            cur_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            cur_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()

            lift_pos = cur_pos.clone()
            lift_pos[2] += 1.0   # z(위쪽)로 10cm, up_axis = Z 이므로 그대로 +z

            indy7_arm.motion_gen.linear_speed = 0.1
            indy7_arm.move_to_cartesian(lift_pos, cur_rot)  # 회전은 그대로 유지(cur_rot)
            state_entered = True
            stable_counter = 0
            prev_ee_pos = None

        # 🔑 JOINT 모드가 아니라 IK(Cartesian) 모드이므로 motion_gen.active를 봐야 함
        if not indy7_arm.motion_gen.active:
            prev_ee_pos, is_stable = check_ee_stable(indy7_arm, prev_ee_pos)
            stable_counter = stable_counter + 1 if is_stable else 0
            if stable_counter >= STABLE_FRAMES_REQUIRED:
                print("[PICK_UP] 완료")
                state = "DONE"
                state_entered = False
                indy7_arm.motion_gen.linear_speed = 5.0
                indy7_arm.detach()


    elif state == "DONE":
        if not state_entered:
            indy7_arm.control_mode = "IK"
            indy7_arm.gym.refresh_rigid_body_state_tensor(indy7_arm.sim)
            indy7_arm.target_pos = indy7_arm.rb_states[indy7_arm.ee_index_global, 0:3].clone()
            indy7_arm.target_rot = indy7_arm.rb_states[indy7_arm.ee_index_global, 3:7].clone()
            state_entered = True

    indy7_arm.process_keyboard_input(events)
    indy7_arm.step()
    indy7_arm.update_attachment()
    indy7_arm.draw_target_marker()
    indy7_arm.draw_ee_marker()

    # 상자 내부 가상 다유도 조인트 데이터 행렬 및 와이어프레임 디버그 렌더
    box_manager1.update_joints()
    box_manager1.draw_debug_visuals(viewer)

    # 물리 엔진 파이프라인 전진 및 프레임 록 동기화
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    frame_count += 1

# 루프 탈출 시 안전하게 메모리 파괴
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("[시스템 종료] Isaac Gym 시뮬레이션 세션이 정상적으로 해제되었습니다.")