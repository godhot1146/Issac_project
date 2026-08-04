import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller import FrankaController
from low_amr_controller_v1 import LowAMR
from lifting_amr_controller_v1 import LiftingAMR

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
# 3. [10m x 8.5m 공장 레이아웃 변형] 에셋 규격 및 가상 맵 정의
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# 새로운 직사각형 공장 사양 (10m x 8.5m)
room_x = 10.0         # 가로 크기 (X축)
room_y = 8.5          # 세로 크기 (Y축)
wall_height = 4.0    # 공장 층고 (10m)
wall_thickness = 0.2  # 외벽 두께 (20cm)

# 방 크기에 맞는 바닥 및 벽 에셋 생성
floor_asset = gym.create_box(sim, room_x, room_y, 0.05, env_opts)

# wall_x: Y축 방향으로 길게 뻗은 벽 (X축 경계면에 배치됨, 길이는 room_y)
wall_x = gym.create_box(sim, wall_thickness, room_y, wall_height, env_opts)
# wall_y: X축 방향으로 길게 뻗은 벽 (Y축 경계면에 배치됨, 길이는 room_x)
wall_y = gym.create_box(sim, room_x, wall_thickness, wall_height, env_opts)

pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height, env_opts) # 코너 H빔 기둥

# =================================================================
# 4. 오픈 팩토리 환경 생성 및 외벽/기둥 배치 (Actors)
# =================================================================
# 환경 영역(Bounding Box) 정의도 변경된 크기에 맞춤
env = gym.create_env(sim, gymapi.Vec3(-room_x/2, -room_y/2, 0), gymapi.Vec3(room_x/2, room_y/2, wall_height), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# 직사각형 외곽 경계벽 스폰 위치 계산
half_x = room_x / 2
half_y = room_y / 2

w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_x, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_x, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_y, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_y, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
    gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

# 공장 프레임 유지를 위한 코너 사각 기둥 배치 (X, Y 축 각각 오프셋 적용)
p_offset_x = half_x - 0.2
p_offset_y = half_y - 0.2
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset_x, p_offset_x]:
    for y in [-p_offset_y, p_offset_y]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)), f"corner_pillar_{x}_{y}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# 외벽 보강 지지 기둥 정렬 (Y축 벽면을 따라 일정 간격으로 X축 양 끝에 배치)
# 8.5m 길이에 맞춰 적절한 간격(-3.0m ~ 3.0m 사이 2.0m 간격)으로 자동 정렬
# pillar_side_positions = np.arange(-3.0, 4.0, 2.0)
# for y_pos in pillar_side_positions:
#     for x_pos in [-p_offset_x, p_offset_x]:
#         p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)), f"side_pillar_{x_pos}_{y_pos}", 0, 0)
#         gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# =============================================================================================
# --- 로봇 인스턴스 스폰 --- 
# =============================================================================================
box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

shelf_opts = gymapi.AssetOptions()
shelf_opts.fix_base_link = False
shelf_opts.density = 100.0
cargo_shelf_asset = gym.load_asset(sim, asset_root, "urdf/cargo_shelf/cargo_shelf_test.urdf", shelf_opts)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v1/low_amr_v1.urdf", low_opts)
# ----------
tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_asset = gym.load_asset(sim, asset_root, "urdf/lifting_amr/v2/lifting_amr_v2_2.urdf", tote_opts)
# ----------
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
vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper.urdf", vacuum_gripper_opts)
# ---------- spawn ----------
# shelf_handle_1 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(1.0, 0.8, 0.1)), "cargo_shelf", -1, 0)
# shelf_handle_2 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(2.5, 0.8, 0.1)), "cargo_shelf", -1, 0)
# shelf_handle_3 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(4.0, 0.8, 0.1)), "cargo_shelf", -1, 0)
# shelf_handle_4 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(1.0, -0.8, 0.1)), "cargo_shelf", -1, 0)
# shelf_handle_5 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(2.5, -0.8, 0.1)), "cargo_shelf", -1, 0)
# shelf_handle_6 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(4.0, -0.8, 0.1)), "cargo_shelf", -1, 0)

low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(-0.5, -2.5, 1.0)), "low_asset", -1, 1)
tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(-0.5, 2.42, 0.3)), "tote_asset", -1, 1)
franka_handle = gym.create_actor(env, vacuum_gripper_asset, gymapi.Transform(p=gymapi.Vec3(-2.5, -2.5, 0.1)), "vacuum_gripper_asset", -1, 0)

# =================================================================
# 🏭 [레이아웃 설계] 지정된 6개 선반 배치 및 URDF 맞춤형 박스 적재
# =================================================================

# 1. 박스 에셋 생성 옵션 (튐 방지 및 고마찰 세팅 유지)
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False 
box_opts.angular_damping = 3.0  
box_opts.linear_damping = 3.0   

box_w, box_l, box_h = 0.55, 0.35, 0.3
box_asset = gym.create_box(sim, box_w, box_l, box_h, box_opts)

# 2. 💡 [수정] 선반 사이 간격 확장 배치 좌표 정의
# X축 간격을 1.5m -> 2.0m로 확장 / Y축 중앙 통로 폭을 1.6m -> 2.4m로 확장
shelf_positions = [
    [1.0,  1.2, 0.1],  # shelf_1
    [3.0,  1.2, 0.1],  # shelf_2
    [1.0, -1.2, 0.1],  # shelf_4 🌟 (LowAMR 진입 대상 - Y좌표가 -1.2로 변경됨)
    [3.0, -1.2, 0.1],  # shelf_5
]

cardboard_color = gymapi.Vec3(0.76, 0.60, 0.42)
box_yaw_angle = math.pi / 2.0
box_rotation = gymapi.Quat.from_euler_zyx(0.0, 0.0, box_yaw_angle)

for idx, pos in enumerate(shelf_positions):
    shelf_num = idx + 1

    # 선반 확장 배치 적용
    pose = gymapi.Transform(p=gymapi.Vec3(pos[0], pos[1], pos[2]))
    s_handle = gym.create_actor(env, cargo_shelf_asset, pose, f"cargo_shelf_{shelf_num}", -1, 0)

    s_props = gym.get_actor_rigid_shape_properties(env, s_handle)
    for s in s_props:
        s.filter = 2
        s.friction = 5.0  
    gym.set_actor_rigid_shape_properties(env, s_handle, s_props)

    # 1층 상판 높이 규격 (0.40m)
    rack_z_levels = [0.40, 1.15, 2.05]
    
    # 박스 간 유격 오프셋
    box_offsets = [
        (-0.30, -0.35),  
        (-0.30,  0.35),  
        ( 0.30, -0.35),  
        ( 0.30,  0.35)   
    ]

    # 4번 선반 1층 완전 비우기 필터 (z_level=0.40 동기화)
    exclude_boxes = [
        (4, 0.40, 0), (4, 0.40, 1), (4, 0.40, 2), (4, 0.40, 3),
        (1, 0.40, 0), (1, 0.40, 1), 
        (3, 2.05, 3), 
        (1, 1.15, 3)   
    ]

    for z_level in rack_z_levels:
        for b_idx, (dx, dy) in enumerate(box_offsets):

            if (shelf_num, z_level, b_idx) in exclude_boxes:
                print(f"[안내] 제외 목록 발견하여 패스 -> 선반{shelf_num} / {z_level}m층 / 박스{b_idx}")
                continue 
            
            rand_x = dx + np.random.uniform(-0.0005, 0.0005)
            rand_y = dy + np.random.uniform(-0.0005, 0.0005)
            
            # 침투 방지 및 안착 마진 최적화 고도 계산
            z_spawn = pos[2] + z_level + 0.02 + (box_h / 2) + 0.025

            b_pose = gymapi.Transform(p=gymapi.Vec3(pos[0] + rand_x, pos[1] + rand_y, z_spawn), r=box_rotation)

            b_name = f"cargo_box_shelf{shelf_num}_z{z_level}_b{b_idx}"
            b_handle = gym.create_actor(env, box_asset, b_pose, b_name, -1, 0)
            gym.set_rigid_body_color(env, b_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cardboard_color)

            # -----------------------------------------------------------
            # 💡 [추가] 박스의 무게(Mass)를 직접 꽂아 넣는 로직
            # -----------------------------------------------------------
            body_props = gym.get_actor_rigid_body_properties(env, b_handle)
            
            # 박스의 무게를 15.0kg으로 강제 지정 (원하는 숫자로 변경 가능)
            target_mass = 0.5 
            body_props[0].mass = target_mass
            
            # (선택 사항) 무게가 바뀐 만큼 관성 모멘트(Inertia)도 물리 엔진이 자동 계산하도록 업데이트
            # 상자 형태의 관성 모멘트 공식 적용: I = (1/12) * M * (w^2 + h^2) 등
            w, l, h = box_w, box_l, box_h
            body_props[0].inertia.x.x = (1.0 / 12.0) * target_mass * (l**2 + h**2)
            body_props[0].inertia.y.y = (1.0 / 12.0) * target_mass * (w**2 + h**2)
            body_props[0].inertia.z.z = (1.0 / 12.0) * target_mass * (w**2 + l**2)
            
            # 변경한 무게 프로퍼티를 액터에 반영
            gym.set_actor_rigid_body_properties(env, b_handle, body_props)
            # -----------------------------------------------------------

            b_props = gym.get_actor_rigid_shape_properties(env, b_handle)
            for b in b_props: 
                b.filter = 1       
                b.friction = 1.0  # 누적 미끄러짐 방지 초고마찰 세팅
                b.restitution = 0.0 
            gym.set_actor_rigid_shape_properties(env, b_handle, b_props)

# ===============================
##### low amr 설정 #####
# ===============================

low_amr = LowAMR(gym, sim, env, low_handle)

waypoint_list = [
    (-1.0, 0.0),    # 1번 목적지
    (2.0, 2.0),    # 2번 목적지
    (-1.5, 3.0),   # 3번 목적지
    (0.0, 0.0)     # 복귀
]
current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 0.2          # 초기 목표 높이를 URDF 한계인 최고점(0.2m)으로 변경
LIFT_THRESHOLD = 0.005     # 가동 범위가 좁으므로 도달 공차를 5mm로 정밀화

# =============================================================================================
# --- LowAMR FSM 상태 정의 및 초기화 --- 
# =============================================================================================
shelf_target_pos = (1.0, -1.2)
delivery_pos     = (-0.5, 1.0)

# 상태 제어 관련 타겟 변수 동기화
LIFT_TARGET_POS = 0.16  # 물리 상승 임계 상한선 타겟값
LIFT_THRESHOLD = 0.005   # 도달 확인 오차 공차 (5mm)

# ===============================
##### lifting amr 설정 #####
# ===============================

lifting_amr = LiftingAMR(gym, sim, env, tote_handle)

# # 1. 💡 [물리 차단 해제] 주행 속도 버퍼 인젝션이 정상 작동하도록 내부 액추에이터 기능 상향 개방
# lifting_amr.configure_actuators(damping=3000.0, max_effort=250000.0)

# # 2. 💡 [핵심 수정] 하드코딩 비트마스크를 지우고, 상자 및 선반과 완벽하게 부딪히도록 전역 콜리전 필터 동기화
# tote_shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
# tote_rigid_body_names = gym.get_actor_rigid_body_names(env, tote_handle)
# num_tote_bodies = gym.get_actor_rigid_body_count(env, tote_handle)

# for body_idx in range(num_tote_bodies):
#     body_name = tote_rigid_body_names[body_idx]
#     shape_range = gym.get_actor_rigid_body_shape_indices(env, tote_handle)[body_idx]
    
#     for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
#         # 💡 [필터 패치] 무조건 모든 액터(상자, 선반, 차체)와 고체 콘택트(Contact)를 맺도록 
#         # 비트마스크 필터를 0으로 초기화하여 충돌을 상시 개방합니다. (Isaac Gym 물리 공식 규격)
#         tote_shape_props[shape_idx].filter = 0 
        
#         # 💡 [마찰력 버스트] 포크 슬라이더와 양측 클로가 상자를 집어 올릴 때 
#         # 미끄러져 튀어 나가지 않도록 접촉 마찰 계수를 극한으로 튜닝합니다.
#         if "fork" in body_name or "claw" in body_name or "slide" in body_name:
#             tote_shape_props[shape_idx].friction = 15.0          # 초고마찰력 주입
#             tote_shape_props[shape_idx].rolling_friction = 0.5   
#             tote_shape_props[shape_idx].restitution = 0.0        # 반발 탄성 제로 (채터링 방지)
#         else:
#             # 바퀴 및 구동 섀시용 마찰력
#             tote_shape_props[shape_idx].friction = 4.0
#             tote_shape_props[shape_idx].rolling_friction = 0.1

# gym.set_actor_rigid_shape_properties(env, tote_handle, tote_shape_props)

waypoint_list = [
    (-1.0, 0.0),    # 1번 목적지
    (2.0, 2.0),    # 2번 목적지
    (-1.5, 3.0),   # 3번 목적지
    (0.0, 0.0)     # 복귀
]
current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 2.5          # 초기 목표 높이: 최고점 (1.8m)
LIFT_THRESHOLD = 0.03      # 높이 도달 인정 공차 (3cm)

rot_state = "CCW"             # 초기 상태: 반시계 방향(좌회전) 상승
rot_target = 1.5708           # 초기 목표 각도: 최대 상한선 (+90도)
ROT_THRESHOLD = 0.03          # 각도 도달 인정 공차 (약 1.7도)

fork_state = "EXTEND"       # 초기 상태: 확장
fork_target = 0.7          # 초기 목표 길이: 최대 상한선 (0.7m)
FORK_THRESHOLD = 0.02      # 도달 인정 공차 (2cm)

claw_state = "GRIP"         # 초기 상태: 오므리기
claw_target = 0.0           # 테스트 목표 각도 (라디안)
CLAW_THRESHOLD = 0.02       # 도달 인정 공차 (약 1.1도)

LIFT_AMR_STATE_IDLE             = 10  # 초기 대기 상태 (LowAMR 완료 신호 대기)

# [선반 상자 인양 섹션]
LIFT_AMR_STATE_MOVE_TO_SHELF    = 11  # 1. 선반으로 위치 이동
LIFT_AMR_STATE_LIFT_TO_SHELF    = 12  # 2. 선반으로 높이 이동
LIFT_AMR_STATE_ROT_TO_BOX       = 13  # 3. 선반으로 트레이 회전
LIFT_AMR_STATE_ARM_EXTEND       = 14  # 4. 상자로 트레이 팔 확장
LIFT_AMR_STATE_CLAW_CLOSE       = 15  # 5. 트레이 claw 닫기 (상자 그랩)
LIFT_AMR_STATE_ARM_RETRACT      = 16  # 6. 트레이 팔 수축 (상자 회수)

# [목표 선반 하역 섹션]
LIFT_AMR_STATE_MOVE_TO_TARGET   = 17  # 7. 목표 선반 위치 이동
LIFT_AMR_STATE_LIFT_TO_TARGET   = 18  # 8. 목표 선반 높이 이동
LIFT_AMR_STATE_ROT_TO_TARGET    = 19  # 9. 목표 선반으로 트레이 회전
LIFT_AMR_STATE_ARM_EXTEND_TGT   = 20  # 10. 목표 선반에 트레이 팔 확장
LIFT_AMR_STATE_CLAW_OPEN        = 21  # 11. 트레이 claw 열기 (상자 방출)
LIFT_AMR_STATE_ARM_RETRACT_TGT  = 22  # 12. 트레이 팔 수축 (하역 완료 및 복귀)

LIFT_AMR_STATE_TASK_COMPLETE    = 30  # 모든 공정 완수 및 정지
LIFT_AMR_STATE_MANUAL_TUNING     = 100 # 수동 조정 예외 모드

# 초기 시작 상태 세팅 (LowAMR 주행 시작 시 대기)
tote_current_state = LIFT_AMR_STATE_IDLE

current_tote_lift   = 0.6 if tote_current_state >= LIFT_AMR_STATE_MOVE_TO_SHELF else 0.0
current_tote_rot    = 0.0
current_tote_fork   = 0.0
current_tote_claw   = -0.8

# 현재 tote_handle 스폰 시작 좌표 p=gymapi.Vec3(-0.5, 2.5, 1.0) 기준
# -x 축 방향으로 부드럽게 후진 이동시킬 목표 타겟 좌표 지정 (예: X축 기준 -0.5m 에서 -1.5m 지점으로 후진)
tote_target_x = -1.2
tote_target_y = 2.45

tote_dest_target_x = 0.85
tote_dest_target_y = 2.45

# ===============================
##### franka 설정 #####
# ===============================
franka_ctrl = FrankaController(gym, sim, env, franka_handle, pump_link_name="cobot_pump", scale=1.6)

# ===============================
##### simulation part #####
# ===============================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "tote_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "tote_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "tote_next_stage")

# 💡 [추가] LowAMR 마지막 단계 조향을 위한 키보드 이벤트 구독
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "low_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "low_right")

# 실시간 수동 주행 속도 버퍼 변수
manual_linear_cmd = 0.0
manual_angular_cmd = 0.0  # 💡 LowAMR 수동 회전 속도 버퍼 추가

frame_count = 0
state_start_frame = None

low_amr.set_state(0)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # 💡 [키보드 이벤트 핸들러 확장]
    for event in gym.query_viewer_action_events(viewer):
        if tote_current_state == LIFT_AMR_STATE_MANUAL_TUNING:
            if event.value > 0:  
                if event.action == "tote_forward":     manual_linear_cmd = 0.4
                elif event.action == "tote_backward":  manual_linear_cmd = -0.4
                elif event.action == "tote_next_stage":
                    print("\n[수동 제어 완료] SPACE 감지 ➡️ 자동 인양 단계 진입")
                    # lifting_amr.set_twist_velocity(0.0, 0.0) 
                    # tote_current_state = LIFT_AMR_STATE_ARM_EXTEND
            elif event.value == 0:  
                if event.action in ["tote_forward", "tote_backward"]:
                    manual_linear_cmd = 0.0

        # LowAMR 수동 조향 키 맵 수신 (FSM 내부적으로 TASK_COMPLETE 옵션 상태일 때 반영됨)
        if event.value > 0:  
            if event.action == "low_left":    manual_angular_cmd = 1.0   
            elif event.action == "low_right": manual_angular_cmd = -1.0  
        elif event.value == 0:  
            if event.action in ["low_left", "low_right"]:
                manual_angular_cmd = 0.0

    # ===============================

    # -----------------------------------------------------------------
    # 🤖 LowAMR FSM 시퀀스 제어 루프
    # -----------------------------------------------------------------
    # 실시간 모바일 섀시 관절 포지션 동기화 데이터 직접 취득
    
    tote_trigger = low_amr.lift_and_locate_fsm(
        frame_count=frame_count,
        target_shelf=shelf_target_pos,
        delivery_pos=delivery_pos,
        manual_angular_cmd=manual_angular_cmd,
        use_manual_steering=True # 💡 수동 조향 옵션 제어 토글 플래그
    )

    if tote_trigger and tote_current_state == LIFT_AMR_STATE_IDLE: 
        tote_current_state = LIFT_AMR_STATE_MOVE_TO_SHELF
        print("\n[LiftingAMR 발크 신호 수신] ➡️ 1단계 선반 이동 시퀀스 시동!")

    # -----------------------------------------------------------------
    # 🤖 2. LiftingAMR 12단계 FSM 시퀀스 제어 루프 (뼈대 프레임)
    # -----------------------------------------------------------------
    # [초기 상태] LowAMR이 작업을 끝내고 신호를 줄 때까지 제자리 대기
    if tote_current_state == LIFT_AMR_STATE_IDLE:
        lifting_amr.set_twist_velocity(0.0, 0.0)

    # 1. 선반으로 위치 이동
    elif tote_current_state == LIFT_AMR_STATE_MOVE_TO_SHELF:
        tote_arrived = lifting_amr.move_to_point(tote_target_x, tote_target_y, backward=True)

        if tote_arrived: 
            print("[LiftingAMR FSM] 1단계 완료: 선반 앞 정위치 이동 완료 ➡️ 2단계(높이 제어) 진입")
            tote_current_state = LIFT_AMR_STATE_LIFT_TO_SHELF

    # 2. 선반으로 높이 이동
    elif tote_current_state == LIFT_AMR_STATE_LIFT_TO_SHELF:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 0.61
        
        # 💡 각 조인트 도달 판정(abs(현재-목표) < 공차) 조건문이 False 자리에 들어갈 예정입니다.
        if abs(current_height - 0.6) < 0.01:
            print("[LiftingAMR FSM] 2단계 완료: 선반 목표 고도 도달 완료 ➡️ 3단계(트레이 정렬 회전) 진입")
            tote_current_state = LIFT_AMR_STATE_ROT_TO_BOX

    # 3. 선반으로 트레이 회전
    elif tote_current_state == LIFT_AMR_STATE_ROT_TO_BOX:
        current_rot = lifting_amr.get_shuttle_rotation()
        rot_target = 1.5708
        current_tote_rot = rot_target
        claw_target = 0.8 
        current_tote_claw = claw_target
        
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            print("[LiftingAMR FSM] 3단계 완료: 트레이 셔틀 정렬 회전 완료 ➡️ 4단계(팔 확장 전진) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_EXTEND

    # 4. 상자로 트레이 팔 확장
    elif tote_current_state == LIFT_AMR_STATE_ARM_EXTEND:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.95
        current_tote_fork = fork_target
        
        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 4단계 완료: 포크 슬라이더 확장 완료 ➡️ 5단계(Claw 그랩 닫기) 진입")
            tote_current_state = LIFT_AMR_STATE_CLAW_CLOSE

    # 5. 트레이 claw 닫기
    elif tote_current_state == LIFT_AMR_STATE_CLAW_CLOSE:
        current_left_claw, _ = lifting_amr.get_claw_positions()
        claw_target = -0.8 
        current_tote_claw = claw_target

        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("[LiftingAMR FSM] 5단계 완료: Claw 상자 그랩 완료 ➡️ 6단계(팔 수축 회수) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_RETRACT

    # 6. 트레이 팔 수축
    elif tote_current_state == LIFT_AMR_STATE_ARM_RETRACT:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.0 
        current_tote_fork = fork_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 6단계 완료: 상자 차체 내 회수 완료 ➡️ 7단계(목표 선반 주행) 진입")
            tote_current_state = LIFT_AMR_STATE_MOVE_TO_TARGET

    # 7. 목표 선반 위치 이동
    elif tote_current_state == LIFT_AMR_STATE_MOVE_TO_TARGET:
        # 💡 move_to_point를 쓰지 않고, X축 정렬 및 직선 주행만 독립적으로 수행합니다.
        # 1. X축 이동을 위해 필요한 목표 각도 판별 (최초 1회 고정)
        if lifting_amr.target_yaw_fixed is None:
            curr_p, _ = lifting_amr.get_pose()
            err_x = tote_dest_target_x - curr_p['x']
            lifting_amr.target_yaw_fixed = 0.0 if err_x > 0 else np.pi
            lifting_amr.turn_complete = False

        # 2. 목표 각도로 선회 정렬
        if not lifting_amr.turn_complete:
            if lifting_amr.turn_to_yaw(lifting_amr.target_yaw_fixed):
                lifting_amr.turn_complete = True
        
        # 3. 각도 정렬 완료 시 X축 직선 주행 전개
        else:
            # drive_along_axis는 목적지 공차(5cm) 안으로 들어오면 True를 반환합니다.
            x_arrived = lifting_amr.drive_along_axis(
                axis='X', 
                target_coordinate=tote_dest_target_x, 
                direction="FORWARD"
            )

            if x_arrived: 
                # 💡 다음 단계를 위해 사용했던 주행용 고정 플래그들을 원상복구합니다.
                lifting_amr.target_yaw_fixed = None
                lifting_amr.turn_complete = False
                lifting_amr.grid_move_stage = 0 # 혹시 모를 오동작 방지 리셋

                print("[LiftingAMR FSM] 7단계 완료: X축 하역 목적지 주행 완료 ➡️ 8단계(하역 고도 이동) 진입")
                tote_current_state = LIFT_AMR_STATE_LIFT_TO_TARGET

    # 8. 목표 선반 높이 이동
    elif tote_current_state == LIFT_AMR_STATE_LIFT_TO_TARGET:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 1.0

        if abs(current_height - 1.0) < 0.01:
            print("[LiftingAMR FSM] 8단계 완료: 하역 목표 고도 도달 완료 ➡️ 9단계(하역 방향 회전) 진입")
            tote_current_state = LIFT_AMR_STATE_ROT_TO_TARGET

    # 9. 목표 선반으로 트레이 회전
    elif tote_current_state == LIFT_AMR_STATE_ROT_TO_TARGET:
        
        current_rot = lifting_amr.get_shuttle_rotation()
        rot_target = 1.5708
        current_tote_rot = rot_target
        
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            print("[LiftingAMR FSM] 9단계 완료: 하역 슬롯 정렬 회전 완료 ➡️ 10단계(하역 팔 확장) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_EXTEND_TGT

    # 10. 목표 선반에 트레이 팔 확장
    elif tote_current_state == LIFT_AMR_STATE_ARM_EXTEND_TGT:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.9
        current_tote_fork = fork_target
        claw_target = 0.8 
        current_tote_claw = claw_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 10단계 완료: 하역지 팔 전진 완료 ➡️ 11단계(Claw 개방 안착) 진입")
            tote_current_state = LIFT_AMR_STATE_CLAW_OPEN

    # 11. 트레이 claw 열기
    elif tote_current_state == LIFT_AMR_STATE_CLAW_OPEN:
        current_left_claw, _ = lifting_amr.get_claw_positions()
        claw_target = 0.8 
        current_tote_claw = claw_target

        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("[LiftingAMR FSM] 11단계 완료: Claw 개방 및 상자 안착 완료 ➡️ 12단계(빈 팔 회수 수축) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_RETRACT_TGT

    # 12. 트레이 팔 수축
    elif tote_current_state == LIFT_AMR_STATE_ARM_RETRACT_TGT:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.0 
        current_tote_fork = fork_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("\n================================================================")
            print("[LiftingAMR 공정 완료] 12단계 모든 인양/하역 시퀀스가 안전하게 완수되었습니다.")
            print("================================================================")
            tote_current_state = LIFT_AMR_STATE_TASK_COMPLETE

    # [최종 종료] 완수 후 완전 제동 정지 대기
    elif tote_current_state == LIFT_AMR_STATE_TASK_COMPLETE:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 0

    # [수동 조정] 키보드 이벤트를 통한 미세 디버깅 튜닝 세션
    elif tote_current_state == LIFT_AMR_STATE_MANUAL_TUNING:
        if manual_linear_cmd > 0:
            lifting_amr.set_twist_velocity(linear_x=manual_linear_cmd, angular_z=0.0)
        elif manual_linear_cmd < 0:
            lifting_amr.set_twist_velocity(linear_x=manual_linear_cmd, angular_z=0.0)
        else:
            lifting_amr.set_twist_velocity(0.0, 0.0)

    # 🎯 LiftingAMR 전용 독립 액추에이터 휠/관절 명령 버퍼 일괄 인젝션
    lifting_amr.set_lift_height(current_tote_lift)
    lifting_amr.set_shuttle_rotation(current_tote_rot)
    lifting_amr.set_fork_extension(current_tote_fork)
    lifting_amr.set_claw_grip(current_tote_claw)
    lifting_amr.apply_actuator_commands()
    # ===============================

    for idx, pos in enumerate(shelf_positions):
        # 각 선반 위치 정보로 Isaac Gym Transform 객체 생성
        shelf_transform = gymapi.Transform(
            p=gymapi.Vec3(pos[0], pos[1], pos[2]),
            r=gymapi.Quat(0, 0, 0, 1) # 회전은 정렬 상태 고정
        )
        
        # 선반 위치를 명확히 판별할 수 있도록 하늘색(Cyan) 구체 사용
        # 선반 규격에 맞게 반경을 0.08m 정도로 보기 편하게 설정
        shelf_geom = gymutil.WireframeSphereGeometry(
            radius=0.08, num_lats=6, num_lons=6, color=(0, 1, 1)
        )
        
        # 시뮬레이션 뷰어 화면에 라인 렌더링
        gymutil.draw_lines(shelf_geom, gym, viewer, env, shelf_transform)

    low_transform = gymapi.Transform(
        p=low_amr.get_pose()[0],
        r=low_amr.get_pose()[1] 
    )
    low_geom = gymutil.WireframeSphereGeometry(
        radius=0.1, num_lats=8, num_lons=8, color=(0, 1, 0)
    )
    
    # 시뮬레이션 뷰어 화면에 라인 렌더링
    gymutil.draw_lines(low_geom, gym, viewer, env, low_transform)

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)