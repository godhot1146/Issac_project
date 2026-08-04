import os
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch
from vacuum_controller_v2 import FrankaController
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

fixed_caser_asset = gym.load_asset(sim, asset_root, "urdf/fixed_caser/v1/fixed_caser_v1.urdf", machinery_opts)
fixed_caser_handle = gym.create_actor(env, fixed_caser_asset, gymapi.Transform(p=gymapi.Vec3(-2.05, -0.2, 0.07)), "fixed_caser_asset", -1, 0)

taping_asset = gym.load_asset(sim, asset_root, "urdf/taping/v2/taping_v2.urdf", machinery_opts)
taping_handle = gym.create_actor(env, taping_asset, gymapi.Transform(p=gymapi.Vec3(-1.15, -0.2, 0.3)), "taping_asset", -1, 1)

# ==============================================================================
# Taping 기구부 관절 제어 설정 초기화
# ==============================================================================
tp_dof_props = gym.get_actor_dof_properties(env, taping_handle)
num_tp_dofs = len(tp_dof_props)

fc_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_cutter_joint", gymapi.DOMAIN_ACTOR)
tp_front_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_taping_front_joint", gymapi.DOMAIN_ACTOR)
tp_back_dof_idx = gym.find_actor_dof_index(env, taping_handle, "base_to_taping_back_joint", gymapi.DOMAIN_ACTOR)

print(f"📦 [Taping 기구부 관절 탐색 완료]")
print(f" - Cutter DOF Index: {fc_dof_idx}")
print(f" - Front Taping DOF Index: {tp_front_dof_idx}")
print(f" - Back Taping DOF Index: {tp_back_dof_idx}")
print(f" - 총 유효 DOF 개수: {num_tp_dofs}")

for i in range(num_tp_dofs):
    tp_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    tp_dof_props['stiffness'][i] = 15000.0  
    tp_dof_props['damping'][i] = 150.0

if num_tp_dofs > 0:
    gym.set_actor_dof_properties(env, taping_handle, tp_dof_props)

current_time = 0.0

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4+0.8, 1.5)), "box_asset", -1, 0)
global_box_body_index = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_ENV)

box_manager = CardboardBoxManager(gym, sim, env, asset_root, pose=gymapi.Transform(p=gymapi.Vec3(-2.44, 0.43, 1.5), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0))))

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = False
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v1/conveyor_v1.urdf", conveyor_opts)
conveyor_handle = gym.create_actor(env, conveyor_rack_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 0.0, 0.1)), "conveyor_asset", -1, 0)

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

vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120.urdf", vacuum_gripper_opts)
vacuum_gripper_asset_v2 = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120_box.urdf", vacuum_gripper_opts)

franka_pos = gymapi.Vec3(3.0, 2.0 + 0.4, 1.3)
franka_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))
franka_transform = gymapi.Transform(p=franka_pos, r=franka_rot)

franka_pos_2 = gymapi.Vec3(-1.7, 0.6, 0.1)
franka_rot_2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
franka_transform_2 = gymapi.Transform(p=franka_pos_2, r=franka_rot_2)

franka_handle = gym.create_actor(env, vacuum_gripper_asset, franka_transform, "vacuum_gripper_asset", -1, 0)
franka_handle_2 = gym.create_actor(env, vacuum_gripper_asset_v2, franka_transform_2, "vacuum_gripper_asset", -1, 1)
desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(-2.7, 0.6, 0.0)), f"desk_asset", -1, 0)

pump_index = gym.find_actor_rigid_body_index(env, franka_handle, "cobot_pump", gymapi.DOMAIN_ACTOR)

desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4, 0.3)), f"desk_asset", -1, 0)
desk_handle2 = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 2.0+0.4+0.8, 0.0)), f"desk_asset2", -1, 0)

# --- 컨트롤러 연동 및 스케일 바인딩 ---
franka_ctrl = FrankaController(gym, sim, env, franka_handle, pump_link_name="cobot_pump", scale=1.6)
franka_ctrl_v2 = FrankaController(gym, sim, env, franka_handle_2, pump_link_name="cobot_pump_base", scale=1.2)
franka_ctrl_v2.use_manual = True

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-1.64, -1.64, 1.70), gymapi.Vec3(-1.83, 1.06, 0.41))

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

# 🤖 [1호기] AMR 적재용 FSM 설정 초기화
franka_ctrl.start_step = 0  
franka_ctrl.end_step = 1   
franka_ctrl.pause_duration = 5
robot_state = "MOVE_DOWN_1"    
step_dur = 1                

# 🤖 [2호기] 박스 폴딩(Caser)용 가상 타겟 초기화
robot_state_v2 = "BOX_CATCH_START"

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

franka_ctrl.is_holding = False 

def step_taping_machinery_by_10_deg(gym, env, actor_handle, num_dofs, fc_idx, tf_idx, tb_idx, cutter_dir=None, front_dir=None, back_dir=None):
    """
    Structured Array의 'pos' 필드를 명확히 참조하여 10도씩 가감하는 함수
    """
    # 1. 현재 관절 상태 가져오기
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    
    # 2. 🚨 중요: state[0] 대신 명확하게 'pos' 필드만 추출하여 1차원 float32 배열 생성
    targets = np.array(dof_states['pos'], dtype=np.float32)
    
    delta_10_deg = np.radians(10.0)
    limit_min, limit_max = -1.57, 1.57

    # 3. 각 관절별 가감 연산 (인덱스가 안전한지 재차 확인)
    if fc_idx != -1 and fc_idx < len(targets) and cutter_dir is not None:
        if cutter_dir == "up":   targets[fc_idx] = min(targets[fc_idx] + delta_10_deg, limit_max)
        if cutter_dir == "down": targets[fc_idx] = max(targets[fc_idx] - delta_10_deg, limit_min)

    if tf_idx != -1 and tf_idx < len(targets) and front_dir is not None:
        if front_dir == "up":   targets[tf_idx] = min(targets[tf_idx] + delta_10_deg, limit_max)
        if front_dir == "down": targets[tf_idx] = max(targets[tf_idx] - delta_10_deg, limit_min)

    # 🎯 이제 인덱스 1번인 Back 관절의 데이터 위치를 정확히 찔러 넣습니다.
    if tb_idx != -1 and tb_idx < len(targets) and back_dir is not None:
        if back_dir == "up":   targets[tb_idx] = min(targets[tb_idx] + delta_10_deg, limit_max)
        if back_dir == "down": targets[tb_idx] = max(targets[tb_idx] - delta_10_deg, limit_min)

    # 4. 물리 엔진에 최종 배열 주입 (3개 축 전체 개수만큼인 데이터가 한 번에 들어가야 함)
    gym.set_actor_dof_position_targets(env, actor_handle, targets)

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
                    
            elif event.action == "select_prev_joint":
                step_taping_machinery_by_10_deg(gym, env, taping_handle, num_tp_dofs, fc_dof_idx, tp_front_dof_idx, tp_back_dof_idx, cutter_dir="up", front_dir="up", back_dir="down")
                print("Cutter 관절 +10도 회전 명령")
            elif event.action == "select_next_joint":
                step_taping_machinery_by_10_deg(gym, env, taping_handle, num_tp_dofs, fc_dof_idx, tp_front_dof_idx, tp_back_dof_idx, cutter_dir="down", front_dir="down", back_dir="up")
                print("Cutter 관절 -10도 회전 명령")
            elif event.action == "reset_process":
                is_active = False
                robot_state = "MOVE_DOWN_1"
                robot_state_v2 = "BOX_CATCH_START"
                franka_ctrl.return_to_ready()
                print("[시스템 알림] 모든 시퀀스를 초기화 상태로 복구합니다.")
            elif event.action == "box_stage_0": box_manager.stage_0_unfolded_flat()
            elif event.action == "box_stage_1": box_manager.stage_1_fold_sides()
            elif event.action == "box_stage_2": box_manager.stage_2_close_bottom()
            elif event.action == "box_stage_3": box_manager.stage_3_close_top()

    gym.clear_lines(viewer)

    # 2호기 로봇 자동 보간 제어 시퀀스 (Caser 박스 폴딩 파트)
    if is_active and robot_state_v2 == "BOX_CATCH_START":
        if franka_ctrl_v2.box_catch_sequence(step_duration=step_dur):
            robot_state_v2 = "BOX_CATCH_ROTATE"
    elif robot_state_v2 == "BOX_CATCH_ROTATE":
        if franka_ctrl_v2.box_rotate_sequence(step_duration=step_dur):
            franka_ctrl_v2.attach(box_manager.handle, distance_threshold=0.2)
            robot_state_v2 = "BOX_FOLD_ROTATE"
    elif robot_state_v2 == "BOX_FOLD_ROTATE":
        franka_ctrl_v2.current_speed_gain = 0.9
        is_done, should_fold = franka_ctrl_v2.box_foldside_sequence(step_duration=step_dur)
        if should_fold:
            box_manager.stage_1_fold_sides()
        if is_done:
            franka_ctrl_v2.current_speed_gain = 1.2
            robot_state_v2 = "BOX_PLACE1"

    elif robot_state_v2 == "BOX_PLACE1":
        if franka_ctrl_v2.box_place1_sequence(step_duration=step_dur):
            box_manager.lock_joint("joint_right_to_front")
            box_manager.lock_joint("joint_right_to_back")
            box_manager.lock_joint("joint_front_to_left")
            robot_state_v2 = "BOX_PLACE2"

    elif robot_state_v2 == "BOX_PLACE2":
        is_done, new_step = franka_ctrl_v2.box_place2_sequence(step_duration=step_dur)
        if new_step is not None:
            print(f"🚀 [알림] BOX_PLACE2 시퀀스의 {new_step}번째 스텝이 시작되었습니다!")
            
            # 특정 스텝 시작 시점에 상자 제어나 출력을 분기하고 싶다면:
            if new_step == 1:
                franka_ctrl_v2.current_speed_gain = 0.5
                box_manager.stage_2_fold_back_to_down1()
            elif new_step == 2:
                franka_ctrl_v2.current_speed_gain = 0.9
                box_manager.stage_2_fold_back_to_down2()
            elif new_step == 3:
                franka_ctrl_v2.current_speed_gain = 1.2
                pass
            elif new_step == 5:
                pass

        if is_done:
            print("✅ BOX_PLACE2 모든 시퀀스 완료!")
            box_manager.lock_joint("joint_back_to_down")
            box_manager.lock_joint("joint_front_to_down")
            robot_state_v2 = "BOX_MOVE"

    elif robot_state_v2 == "BOX_MOVE":
        is_done, new_step = franka_ctrl_v2.box_move_sequence(step_duration=step_dur)
        if new_step is not None:            
            if new_step == 1:
                print("박스 접기")
                box_manager.close_side_bottom()
            if new_step == 2:
                box_manager.close_side_bottom2()
                # box_manager.lock_joint("joint_right_to_down")
                # box_manager.lock_joint("joint_left_to_down")
            if new_step == 5:
                step_taping_machinery_by_10_deg(gym, env, taping_handle, num_tp_dofs, fc_dof_idx, tp_front_dof_idx, tp_back_dof_idx, cutter_dir="up", front_dir="up", back_dir="down")
            if new_step == 10:
                step_taping_machinery_by_10_deg(gym, env, taping_handle, num_tp_dofs, fc_dof_idx, tp_front_dof_idx, tp_back_dof_idx, cutter_dir="down", front_dir="down", back_dir="up")
        if is_done:
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
    franka_ctrl.draw_ee_debug_sphere(viewer, radius=0.04, color=(1.0, 0.0, 0.0))
    franka_ctrl_v2.draw_ee_debug_sphere(viewer, radius=0.04, color=(0.0, 1.0, 0.0))

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