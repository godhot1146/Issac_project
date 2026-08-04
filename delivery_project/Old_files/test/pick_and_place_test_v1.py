import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController
from cardboard_box_manager import CardboardBoxManager
from low_amr_controller_v1 import LowAMR

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

machinery_opts = gymapi.AssetOptions()
machinery_opts.fix_base_link = True           
machinery_opts.density = 100.0
machinery_opts.armature = 0.01                
machinery_opts.linear_damping = 0.5
machinery_opts.angular_damping = 0.5
machinery_opts.disable_gravity = True         

current_time = 0.0

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

box_manager = CardboardBoxManager(gym, sim, env, asset_root, pose=gymapi.Transform(p=gymapi.Vec3(-0.0, -0.4+0.05, 0.9), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0))))

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = True
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v2/conveyor_v2.urdf", conveyor_opts)

conveyor_pos = gymapi.Vec3(0.5, -0.5, 0.0)
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

vacuum_gripper_opts = gymapi.AssetOptions()
vacuum_gripper_opts.fix_base_link, vacuum_gripper_opts.flip_visual_attachments = True, True
vacuum_gripper_opts.armature = 0.01
vacuum_gripper_opts.thickness = 0.001
vacuum_gripper_opts.linear_damping = 0.0
vacuum_gripper_opts.angular_damping = 0.0
vacuum_gripper_opts.override_com = True
vacuum_gripper_opts.override_inertia = True

vacuum_gripper_asset_v2 = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120_v3.urdf", vacuum_gripper_opts)

franka_pos_2 = gymapi.Vec3(0.0, 0.0, 0.6)
franka_rot_2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
franka_transform_2 = gymapi.Transform(p=franka_pos_2, r=franka_rot_2)

franka_handle_2 = gym.create_actor(env, vacuum_gripper_asset_v2, franka_transform_2, "vacuum_gripper_asset", -1, 1)

desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0, 0.0)), f"desk_asset", -1, 0)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v1/low_amr_v1.urdf", low_opts)
low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 1.0, 0.2)), "low_asset", -1, 1)

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/low_move_rack/low_move_rack.urdf", move_rack_opts)
move_rack_handle = gym.create_actor(env, move_rack_asset, gymapi.Transform(p=gymapi.Vec3(-2.0, 1.0, 0.05)), "move_rack_asset", -1, 0)

dummy_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/dummy.urdf", move_rack_opts)
dummy_handle1 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0-0.4, 0.5-0.02, 1.2+0.1)), "dummy_asset1", -1, 0)
dummy_handle2 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0+0.15, 0.5, 1.2+0.1)), "dummy_asset2", -1, 0)
dummy_handle3 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0-0.15, 0.5, 1.2+0.1)), "dummy_asset3", -1, 0)
dummy_handle4 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0+0.4, 0.5, 1.2+0.1)), "dummy_asset4", -1, 0)
# dummy_handle5 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0-0.4, 0.7, 1.2+0.1)), "dummy_asset5", -1, 0)
# dummy_handle6 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0+0.15, 0.7, 1.2+0.1)), "dummy_asset6", -1, 0)
# dummy_handle7 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0-0.15, 0.7, 1.2+0.1)), "dummy_asset7", -1, 0)
dummy_handle8 = gym.create_actor(env, dummy_asset, gymapi.Transform(p=gymapi.Vec3(-2.0+0.4, 0.7, 1.2+0.1)), "dummy_asset8", -1, 0)

# --- 컨트롤러 연동 및 스케일 바인딩 ---
franka_ctrl_v2 = FrankaController(gym, sim, env, franka_handle_2, pump_link_name="cobot_pump_base", scale=1.2)
franka_ctrl_v2.use_manual = True

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-1.64, -1.64, 1.70), gymapi.Vec3(-1.83, 1.06, 0.41))

# ===============================
##### low amr 설정 #####
# ===============================

low_amr = LowAMR(gym, sim, env, low_handle)

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

low_amr.set_state(0)

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

rotation_time_accumulator = 0.0
is_active = False  
is_loaded = False
frame_count = 0

# 🤖 [2호기] 초기화
robot_state_v2 = "CATCH_START"
step_dur = 1    
target_relative_xyz = [0.7706, -0.0008, 0.0065] # 1.2배 스케일링

# 변수 및 컨베이어 물리 기초 데이터 설정
L = 3.5                 
P = 128                 
v = 0.5                
dt = 1.0 / 60.0         
pixels_per_frame = (v * dt * P) / L  
accumulated_pixel_shift = 0.0
prev_box_z = 999.0      
is_landed = False       
is_grabed = False
animation_speed_frames = 5  

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

# ==============================================================================
# 🚀 메인 시뮬레이션 루프 진입
# ==============================================================================
while not gym.query_viewer_has_closed(viewer):
    
    # ① 키보드 입력 핸들링 및 다이렉트 호출 분기
    for event in gym.query_viewer_action_events(viewer):
        is_pressed = (event.value > 0)
        
        if event.action in key_states:
            key_states[event.action] = is_pressed

        # -------------------------------------------------------------
        # 💡 [예전 코드의 핵심 구조]
        # 키를 누르는 순간(event.value > 0)에만 컨트롤러로 이벤트를 넘겨줍니다!
        # -------------------------------------------------------------
        if franka_ctrl_v2.use_manual and event.value > 0:
            franka_ctrl_v2.handle_keyboard_event(event.action)
        # -------------------------------------------------------------

        if box_manager.use_manual and event.value > 0:
             box_manager.handle_keyboard_event(event.action)

        # 일반 키 이벤트 처리 (여기는 원래 코드와 동일)
        if event.value > 0: 
            if event.action == "start_process":
                is_active = True
                
                rel_pos, zyx_deg = franka_ctrl_v2.get_end_effector_pose_relative_to_base()
                if rel_pos is not None:
                    # 🎯 현재 클래스에 주입된 scale(1.6 또는 1.2)로 나누어 1배 스케일의 위치 역산
                    # 만약 클래스 멤버 변수가 없다면 직접 고정값 1.2를 대입해도 됩니다.
                    scale_factor = getattr(franka_ctrl_v2, 'scale', 1.2)
                    rel_pos_1x = rel_pos / scale_factor

                    # 위치: X, Y, Z (m) -> 1배 기준으로 축소된 실제 로봇 좌표값
                    print(f"📍 Base-Relative Position (1x) -> X: {rel_pos_1x[0]:.3f}m, Y: {rel_pos_1x[1]:.3f}m, Z: {rel_pos_1x[2]:.3f}m")
                    
                    # 각도: Rot Z, Rot Y, Rot X (deg) -> 각도는 스케일 변동이 없으므로 zyx_deg 그대로 사용
                    print(f"🔄 Base-Relative Rotation -> Rot Z: {zyx_deg[0]:.1f}°, Rot Y: {zyx_deg[1]:.1f}°, Rot X: {zyx_deg[2]:.1f}°")

                    # 💡 RoboDK GUI 좌표 패널과 다이렉트로 대조하고 싶을 때 (mm 단위 환산 팁)
                    # print(f"🤖 RoboDK 입력용(mm) -> X: {rel_pos_1x[0]*1000:.1f}, Y: {rel_pos_1x[1]*1000:.1f}, Z: {rel_pos_1x[2]*1000:.1f}")

                    print(actor_origin(franka_handle_2))
                    print(actor_origin(dummy_handle1)/1.2)
                    print(actor_origin(dummy_handle2)/1.2)

            elif event.action == "reset_process":
                is_active = False
                robot_state_v2 = "CATCH_START"
                print("[시스템 알림] 모든 시퀀스를 초기화 상태로 복구합니다.")
            elif event.action == "box_stage_0": box_manager.stage_0_unfolded_flat()
            elif event.action == "box_stage_1": box_manager.stage_1_fold_sides()
            elif event.action == "box_stage_2": box_manager.stage_2_close_bottom()
            elif event.action == "box_stage_3": box_manager.stage_3_close_top()

    gym.clear_lines(viewer)

    low_amr.lift_and_locate_fsm_v2(
        frame_count=frame_count,
        waypoint_list=waypoint_list
    )

    if is_active and robot_state_v2 == "CATCH_START":
        if franka_ctrl_v2.catch_sequence(step_duration=step_dur):
            franka_ctrl_v2.attach(dummy_handle1)
            robot_state_v2 = "CATCH_START2"

    if is_active and robot_state_v2 == "CATCH_START2":
        if franka_ctrl_v2.catch_sequence2(step_duration=step_dur):
            robot_state_v2 = "MOVE"

    elif robot_state_v2 == "MOVE":
        if franka_ctrl_v2.catch_move_sequence(step_duration=step_dur):
            franka_ctrl_v2.detach()
            robot_state_v2 = "END"


    # 2호기 상향 고강성 전용 실시간 관절 벨로시티 보간기 업데이트
    franka_ctrl_v2.update_joints_interpolation()    
    franka_ctrl_v2.update_snapping_object()

    # 수동 조작 모드 플래그 활성화 시 물리 텐서 데이터 강제 주입 우회
    if franka_ctrl_v2.manual_mode:
        franka_ctrl_v2.apply_manual_targets()

    # 상자 내부 가상 다유도 조인트 데이터 행렬 및 와이어프레임 디버그 렌더
    box_manager.update_joints()
    box_manager.draw_debug_visuals(viewer)
    franka_ctrl_v2.draw_ee_debug_sphere(viewer, radius=0.04, color=(0.0, 1.0, 0.0))
    franka_ctrl_v2.draw_z_rotational_circle(
        viewer, 
        target_relative_xyz, 
        num_spheres=20, 
        sphere_radius=0.04, 
        color=(0.0, 0.5, 1.0)  # 스카이 블루 색상
    )

    draw_actor_origin_debug_sphere(viewer, dummy_handle1)
    draw_actor_origin_debug_sphere(viewer, dummy_handle8)
    draw_actor_origin_debug_sphere(viewer, franka_handle_2)

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