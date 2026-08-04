import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller import FrankaController

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

# ===============================
# --- 로봇 인스턴스 스폰 --- 
# ===============================
box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

shelf_opts = gymapi.AssetOptions()
shelf_opts.fix_base_link = False
shelf_opts.density = 100.0
cargo_shelf_asset = gym.load_asset(sim, asset_root, "urdf/cargo_shelf/cargo_shelf_test.urdf", shelf_opts)

low_amr_opts = gymapi.AssetOptions()
low_amr_opts.fix_base_link = False
low_amr_asset = gym.load_asset(sim, asset_root, "urdf/low_amr/low_amr_edited.urdf", low_amr_opts)

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_asset = gym.load_asset(sim, asset_root, "urdf/tote/whole/test.urdf", tote_opts)

vacuum_gripper_opts = gymapi.AssetOptions()
vacuum_gripper_opts.fix_base_link, vacuum_gripper_opts.flip_visual_attachments = True, True
vacuum_gripper_opts.armature = 0.01
vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper.urdf", vacuum_gripper_opts)

low_amr_handle = gym.create_actor(env, low_amr_asset, gymapi.Transform(p=gymapi.Vec3(0.8, 1.5, 0.1)), "low_amr", -1, 0)
shelf_handle_1 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 1.5, 0.1)), "cargo_shelf", -1, 0)
shelf_handle_2 = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(3.0, -3.5, 0.1)), "cargo_shelf", -1, 0)
franka_handle = gym.create_actor(env, vacuum_gripper_asset, gymapi.Transform(p=gymapi.Vec3(-1.0, -1.0, 0.055)), "vacuum_gripper_asset", -1, 0)
tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(1.0, -3.5, 0.2)), "tote_asset", -1, 0)

# ===============================
##### 리프팅 amr 설정 #####
# ===============================
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

# tote_handle 용 휠 속도 제어 배열 (이미 선언되어 있다면 확인만 하셔도 됩니다)
tote_num_dofs = len(gym.get_actor_dof_properties(env, tote_handle))
tote_velocities = np.zeros(tote_num_dofs, dtype=np.float32)

# -x 방향 후진을 위한 각속도 계산 (예: 초당 -0.2m 이동 대상)
# 바퀴 반지름 0.15m이므로 각속도 = 선속도 / 반지름 -> -0.2 / 0.15 = -1.33 rad/s
tote_backward_speed = 1.33

# 리프팅 amr 디버그 구체 위치관련 설정
dof_props = gym.get_actor_dof_properties(env, tote_handle)
dof_dict = gym.get_actor_dof_dict(env, tote_handle)

left_wheel_idx = dof_dict['left_wheel_joint']
right_wheel_idx = dof_dict['right_wheel_joint']
lift_z_idx = dof_dict['lift_z_joint']
lift_rot_idx = dof_dict['lift_rotation_joint']
fork_ext_idx = dof_dict['fork_extension_joint']
claw_idx = dof_dict['left_claw_joint']      # 왼쪽 문 역할
claw2_idx = dof_dict['right_claw_joint']    # 오른쪽 문 역할

claw1_rb_idx = gym.find_actor_rigid_body_index(env, tote_handle, "left_claw_link", gymapi.DOMAIN_ACTOR)
claw2_rb_idx = gym.find_actor_rigid_body_index(env, tote_handle, "right_claw_link", gymapi.DOMAIN_ACTOR)

TARGET_ROTATION = math.pi / 2

sphere_origin = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(1, 1, 1)) # 원점: 흰색
sphere_x      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(1, 0, 0)) # X축: 빨간색 (Red)
sphere_y      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(0, 1, 0)) # Y축: 초록색 (Green)
sphere_z      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(0, 0, 1)) # Z축: 파란색 (Blue)

# viewer = gym.create_viewer(...) 위아래 적절한 위치에 추가
mast_rb_idx = gym.find_actor_rigid_body_index(env, tote_handle, "mast_link", gymapi.DOMAIN_ACTOR)
print(f"-> Detected mast_link RigidBody Index: {mast_rb_idx}")

# 시각화에 사용할 구체 형상 정의 (반지름 8cm)
sphere_mast_origin = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=12, num_lons=12, color=(1, 1, 0)) # 원점: 노란색
sphere_mast_com    = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=12, num_lons=12, color=(1, 0, 1)) # 무게중심: 자홍색(Magenta)

# 에셋의 전체 강체(Rigid Body) 관성 속성들을 가져옵니다.
mast_link_properties = gym.get_actor_rigid_body_properties(env, tote_handle)

# com 속성 자체가 이미 gymapi.Vec3 (x, y, z 좌표) 객체입니다.
local_com_offset = mast_link_properties[mast_rb_idx].com

# ===============================
##### low amr 설정 ##### 
# ===============================
low_amr_props = gym.get_actor_rigid_shape_properties(env, low_amr_handle)

# 1. 먼저 모든 링크(샤시 및 피동 바퀴 포함)의 마찰력을 낮게 설정 (예: 0.1)
# 피동 바퀴가 미끄러지듯 잘 따라오게 만들기 위함입니다.
for s in low_amr_props:
    s.friction = 0.05          # 미끄러짐 마찰력 (Caster 역할)
    s.rolling_friction = 0.01  # 구름 마찰력

# 2. 구동 바퀴(Drive Wheels)가 위치한 Shape 인덱스만 찾아 마찰력을 높임
# URDF 로드 순서에 따라 달라지므로, 안전하게 구동 바퀴 조인트의 링크 인덱스를 활용합니다.
# 일반적으로 구동 바퀴의 비주얼/충돌 형상은 아래와 같이 인덱싱됩니다.
# (만약 구동 바퀴가 슬립이 난다면 아래 인덱스 번호를 1, 2 등으로 조정해 보세요)
drive_shape_indices = [1, 2] 

for idx in drive_shape_indices:
    if idx < len(low_amr_props):
        low_amr_props[idx].friction = 15.0          # 구동력을 위한 높은 마찰력
        low_amr_props[idx].rolling_friction = 0.5

gym.set_actor_rigid_shape_properties(env, low_amr_handle, low_amr_props)

# --- [추가] DOF 속성 및 구동 바퀴 인덱스 검출 ---
amr_dof_props = gym.get_actor_dof_properties(env, low_amr_handle)
amr_dof_dict = gym.get_actor_dof_dict(env, low_amr_handle)

# URDF에 정의된 구동 바퀴 조인트 이름 매핑
amr_left_idx = amr_dof_dict['left_drive_joint']
amr_right_idx = amr_dof_dict['right_drive_joint']
lift_idx = amr_dof_dict['lift_joint']

# 모든 조인트를 속도 제어 모드로 변경 (구동 바퀴 외 피동 바퀴는 마찰 없이 굴러가도록 세팅)
for i in range(len(amr_dof_props)):
    amr_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_VEL)
    if i in [amr_left_idx, amr_right_idx]:
        amr_dof_props['stiffness'][i] = 0.0
        amr_dof_props['damping'][i] = 600.0   
        amr_dof_props['velocity'][i] = 20.0     
        amr_dof_props['effort'][i] = 20000.0      
    elif i == lift_idx:
        # [수정] 리프트 조인트만 위치 제어 모드(POS)로 타겟팅
        amr_dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
        amr_dof_props['stiffness'][i] = 150000.0  # 무거운 선반을 번쩍 들 수 있는 강력한 강성
        amr_dof_props['damping'][i] = 5000.0       # 출렁임 방지용 댐핑
        amr_dof_props['velocity'][i] = 0.05        # 초당 최대 50cm 상승 한계
        amr_dof_props['effort'][i] = 80000.0      # 최대 토크/힘 제한 대폭 상향
    else:
        amr_dof_props['stiffness'][i] = 0.0
        amr_dof_props['damping'][i] = 1.0     
        amr_dof_props['velocity'][i] = 20.0     
        amr_dof_props['effort'][i] = 2000.0

gym.set_actor_dof_properties(env, low_amr_handle, amr_dof_props)

# low_amr 물리 상수 정의 (URDF 기준 반지름: 0.12m, 바퀴 간격: 0.46 * 2 = 0.92m)
amr_wheel_radius = 0.12
amr_wheel_base = 0.92
amr_velocities = np.zeros(len(amr_dof_props), dtype=np.float32)

# 목표 위치 설정 (X: 3.0, Y: 1.5)
target_pos = np.array([3.0, 1.5])

lift_idx = amr_dof_dict['lift_joint']

# 미션 상태(State) 정의
# 0: 최초 목표지점(3.0, 1.5)으로 주행
# 1: 목표지점 도착 후 리프트 상승 (선반 리프팅)
# 2: 선반을 든 채 복귀지점(1.0, 1.5)으로 주행
# 3: 최종 미션 완료
current_state = 0 

# 리프트 제어 변수
LIFT_TARGET_POS = 0.115  # URDF 상 limit upper가 0.1이므로 최대에 가까운 0.095m까지 상승
lift_velocities = np.zeros(len(amr_dof_props), dtype=np.float32)

# ===============================
##### franka 설정 #####
# ===============================
franka_ctrl = FrankaController(gym, env, franka_handle)

f_props = gym.get_actor_dof_properties(env, franka_handle)
f_props["driveMode"].fill(gymapi.DOF_MODE_POS)
f_props["stiffness"].fill(500.0); f_props["damping"].fill(10.0)
gym.set_actor_dof_properties(env, franka_handle, f_props)

# ===============================
##### simulation part #####
# ===============================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # ===============================
    
    # ==========================================================
    # [추가] low_amr 주행 제어 알고리즘 (P 제어기 기반)
    # ==========================================================
    # 1. 로봇 상태(Pose 및 Joint 상태) 동기화
    rb_states = gym.get_actor_rigid_body_states(env, low_amr_handle, gymapi.STATE_POS)
    amr_pose = rb_states['pose'][0] 
    current_pos = np.array([amr_pose['p']['x'], amr_pose['p']['y']])
    
    # 헤딩각(Yaw) 계산
    qw, qx, qy, qz = amr_pose['r']['w'], amr_pose['r']['x'], amr_pose['r']['y'], amr_pose['r']['z']
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # 현재 조인트 위치(리프트 높이 확인용) 가져오기
    joint_states = gym.get_actor_dof_states(env, low_amr_handle, gymapi.STATE_POS)
    current_lift_pos = joint_states['pos'][lift_idx]

    # 초기 속도 명령 초기화
    linear_vel = 0.0
    angular_vel = 0.0
    lift_vel_cmd = 0.0

    # 2. 상태별 시퀀스 제어 (State Machine)
    if current_state == 0:
        # [State 0] 선반이 있는 (3.0, 1.5) 주행
        target_pos = np.array([3.0, 1.5])
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        if distance > 0.06: # 허용 오차 약 6cm
            target_yaw = math.atan2(pos_error[1], pos_error[0])
            yaw_error = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            
            linear_vel = np.clip(1.5 * distance, -0.8, 0.8)
            angular_vel = np.clip(4.0 * yaw_error, -1.2, 1.2)
        else:
            # 도착 시 정지 후 리프팅 단계(State 1)로 전환
            print("-> [State 0 완료] 선반 하부 도착. 리프팅을 시작합니다.")
            current_state = 1

    elif current_state == 1:
        # [State 1] Prismatic Joint 구동하여 선반 들어 올리기
        lift_error = LIFT_TARGET_POS - current_lift_pos

        if frame_count%60==0:
            print("current_lift_pos ", current_lift_pos)

        
        # 목표 높이와 현재 높이의 차이를 체크
        if lift_error > 0.017: # 오차 1cm 기준
            # [핵심 수정] 속도 명령을 0.04에서 0.3으로 상향하여 눈에 보이게 만듭니다.
            lift_vel_cmd = 0.05 
        else:
            lift_vel_cmd = 0.0
            print(f"-> [State 1 완료] 최종 높이 {current_lift_pos:.3f}m 도달. 복귀 시작!")
            current_state = 2

    elif current_state == 2:
        # [State 2] 회전하지 않고, 바퀴만 반대로 굴려 (1.0, 1.5) 좌표로 후진 주행
        target_pos = np.array([1.0, 1.5])
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        if distance > 0.06:
            # 후진 주행이므로, 목표 각도를 로봇의 정반대 방향(+pi)으로 설정합니다.
            target_yaw = math.atan2(pos_error[1], pos_error[0]) + math.pi
            
            # 각도 오차 범위를 [-pi, pi]로 정규화
            yaw_error = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            
            # [핵심] 후진해야 하므로 선속도(linear_vel) 값을 음수(-)로 클리핑합니다.
            # 거리가 멀수록 뒤로 빠르게 이동하되 최대 후진 속도를 -0.5로 제한합니다.
            linear_vel = np.clip(-1.2 * distance, -0.5, 0.0) 
            
            # 후진 중 일어나는 미세한 방향 이탈을 잡기 위한 각속도 제어
            angular_vel = np.clip(3.5 * yaw_error, -1.0, 1.0)
        else:
            print("-> [State 2 완료] 회전 없이 후진으로 복귀 좌표에 정상 안착했습니다. 미션 종료.")
            current_state = 3

    elif current_state == 3:
        # [State 3] 복귀 좌표 안착 후, 목적지가 있는 Y축 음의 방향(-90도)으로 제자리 회전
        # [수정] math.pi / 2(위쪽) 대신 -math.pi / 2(아래쪽)를 바라보게 합니다.
        target_yaw = -math.pi / 2 
        
        yaw_error = target_yaw - current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        
        if frame_count%60==0:
            print("yaw_error ", yaw_error)

        if abs(yaw_error) > 0.090:
            linear_vel = 0.0
            angular_vel = np.clip(3.5 * yaw_error, -0.8, 0.8)
        else:
            print(f"-> [State 3 완료] 목적지 방향(-90도) 정렬 완료. State 4로 전환합니다.")
            current_state = 4

    elif current_state == 4:
        # [State 4] 새로운 목적지 (1.0, -2.0)로 직진(전진) 주행
        # [수정] 주석의 상태 번호와 변수 일치 (target_pos 확인)
        target_pos = np.array([1.0, -2.0])
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        if distance > 0.06:
            target_yaw = math.atan2(pos_error[1], pos_error[0])
            yaw_error = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            
            linear_vel = np.clip(1.2 * distance, 0.0, 0.6)
            angular_vel = np.clip(4.0 * yaw_error, -1.0, 1.0)
        else:
            print("-> [State 4 완료] 최종 목적지 (1.0, -2.0)에 직선 주행으로 안전하게 도달했습니다.")
            current_state = 5

    # 3. 차동 구동 공식 변환 및 바퀴/리프트 속도 입력 적용
    v_left = (linear_vel - (angular_vel * amr_wheel_base / 2.0)) / amr_wheel_radius
    v_right = (linear_vel + (angular_vel * amr_wheel_base / 2.0)) / amr_wheel_radius

    # 모든 제어 명령을 하나의 속도 배열에 맵핑하여 전송
    # 바퀴는 기존대로 속도 제어 배열에 주입
    amr_velocities[amr_left_idx] = v_left
    amr_velocities[amr_right_idx] = v_right
    # [수정] 리프트 속도 배열에서는 제외 (위치 제어로 넘길 것이므로)
    amr_velocities[lift_idx] = 0.0  
    gym.set_actor_dof_velocity_targets(env, low_amr_handle, amr_velocities)

    # [추가] 리프트 조인트 전용 위치 제어 명령 발송
    # State 1일 때는 목표 높이(0.495)로, 그 외엔 현재 진행할 목표 높이 유지
    lift_targets = np.zeros(len(amr_dof_props), dtype=np.float32)
    if current_state >= 1:
        lift_targets[lift_idx] = LIFT_TARGET_POS
    else:
        lift_targets[lift_idx] = 0.0
    gym.set_actor_dof_position_targets(env, low_amr_handle, lift_targets)

    # ===============================
    # left_wheel_idx와 right_wheel_idx에 음수 속도를 동일하게 주어 뒤로 직진하게 만듭니다.

    if current_state == 5:
        # [State 5] tote_handle을 -x 방향(후진)으로 이동 및 한계값 체크
        
        # 1. tote_handle의 현재 위치(Pose) 상태 가져오기
        tote_rb_states = gym.get_actor_rigid_body_states(env, tote_handle, gymapi.STATE_POS)
        # base_link(보통 0번 또는 1번 인덱스, 여기서는 안전하게 루트 링크인 0번)의 x 좌표 추출
        current_tote_x = tote_rb_states['pose'][0]['p']['x']
        
        if frame_count%60==0:
            print("current_tote_x ", current_tote_x)

        # 2. 이동 한계값 지정 (예: 초기 X가 1.0이므로 0.5m 후진한 0.5를 한계로 지정)
        X_LIMIT = 0.6 
        
        if current_tote_x > X_LIMIT:
            # 한계값에 도달하기 전까지는 계속 후진 속도 주입
            tote_velocities[left_wheel_idx] = tote_backward_speed
            tote_velocities[right_wheel_idx] = tote_backward_speed
        else:
            # 한계값을 넘기면 바퀴를 정지시키고 다음 상태(State 6)로 전환
            tote_velocities[left_wheel_idx] = 0.0
            tote_velocities[right_wheel_idx] = 0.0
            print(f"-> [State 5 완료] tote_handle이 한계값({X_LIMIT}m)에 도달했습니다. (현재 X: {current_tote_x:.3f}m)")
            print("-> State 6으로 전환합니다.")
            current_state = 6
            
        # 계산된 속도를 tote_handle에 적용
        gym.set_actor_dof_velocity_targets(env, tote_handle, tote_velocities)

    elif current_state == 6:
        # [State 6] tote_handle 이동 완료 후 정지 및 다음 미션 대기 상태
        tote_velocities[left_wheel_idx] = 0.0
        tote_velocities[right_wheel_idx] = 0.0
        gym.set_actor_dof_velocity_targets(env, tote_handle, tote_velocities)
        
        # 만약 이 단계에서 low_amr 등 다른 로봇도 멈춰야 한다면 함께 속도를 0으로 제어해 줍니다.
        linear_vel = 0.0
        angular_vel = 0.0
    # 다른 조인트(리프트, 포크 등)는 움직이지 않고 현재 상태를 유지하도록 0.0 처리
    # (단, 위치 제어(POS) 모드인 조인트가 있다면 속도가 아닌 위치 타겟을 주어야 하므로 
    #  여기서는 속도 모드로 기본 세팅된 left_wheel, right_wheel에만 명령을 주입합니다.)
    
    gym.set_actor_dof_velocity_targets(env, tote_handle, tote_velocities)

    # ===============================

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)