import os
import numpy as np
from isaacgym import gymapi, gymutil
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
#b_handle = gym.create_actor(env, box_asset, b_pose, "cargo_box", -1, 0)
#gym.set_rigid_body_color(env, b_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False

tote_asset = gym.load_asset(sim, asset_root, "urdf/lifting_amr/v2/lifting_amr_v2.urdf", tote_opts)
tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.0)), "tote_asset", -1, 1)

lifting_amr = LiftingAMR(gym, sim, env, tote_handle)

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

# =========================================================================

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # =========================================================================
    # 직관적인 Twist 기반 속도 제어 명령 하달
    # =========================================================================
    if current_wp_idx < len(waypoint_list):
        target_x, target_y = waypoint_list[current_wp_idx]
        
        # 함수를 호출하면 매 스텝 목적지로 알아서 찾아가며, 완전히 도달하면 True를 리턴합니다.
        is_arrived = lifting_amr.move_to_point(target_x, target_y)
        
        if is_arrived:
            print(f"★ [{current_wp_idx + 1}번 목적지] ({target_x}, {target_y}) 도달 완료! 다음 좌표로 이동합니다.")
            current_wp_idx += 1
    else:
        # 모든 경유지 순회 완료 시 최종 제자리 대기
        lifting_amr.set_twist_velocity(0.0, 0.0)
	
    # # =========================================================================

    # 1. 현재 셔틀의 실제 높이 모니터링
    current_height = lifting_amr.get_lift_height()
    
    # 2. 상태 머신 전환 조건 검사
    if lift_state == "UP":
        # 최고점(1.8m) 근처에 도달했는지 체크
        if abs(lift_target - current_height) < LIFT_THRESHOLD:
            print("▲ 최고점 도달! 하강 시퀀스로 전환합니다.")
            lift_state = "DOWN"
            lift_target = 0.0  # 목표치를 최저점(0.0m)으로 변경
            
    elif lift_state == "DOWN":
        # 최저점(0.0m) 근처에 도달했는지 체크
        if abs(lift_target - current_height) < LIFT_THRESHOLD:
            print("▼ 최저점 도달! 상승 시퀀스로 전환합니다.")
            lift_state = "UP"
            lift_target = 2.5  # 목표치를 최고점(1.8m)으로 변경

    if frame_count%60==0:
        print("lift_target - current_height: ", lift_target - current_height)

    # 3. 매 스텝 설정된 목표 높이 명령을 POS 제어기에 주입
    #lifting_amr.set_lift_height(lift_target)
    
    # =========================================================================

    # 1. 현재 셔틀의 실제 회전 각도(라디안) 모니터링
    current_rot = lifting_amr.get_shuttle_rotation()
    
    # 2. 상태 머신 전환 조건 검사
    if rot_state == "CCW":
        # 최대 양수 각도(+1.5708 rad) 부근에 도달했는지 체크
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            rot_state = "CW"
            rot_target = -1.5708  # 목표치를 음수 하한선(-90도)으로 변경
            
    elif rot_state == "CW":
        # 최소 음수 각도(-1.5708 rad) 부근에 도달했는지 체크
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            rot_state = "CCW"
            rot_target = 1.5708   # 목표치를 다시 양수 상한선(+90도)으로 변경

    # 3. 매 스텝 설정된 목표 회전 각도 명령을 POS 제어기에 주입
    #lifting_amr.set_shuttle_rotation(rot_target)

    # =========================================================================

    # 1. 현재 포크의 실제 연장 길이 모니터링
    current_fork = lifting_amr.get_fork_extension()
    
    # 2. 상태 머신 전환 조건 검사
    if fork_state == "EXTEND":
        # 최대 확장(0.7m) 근처에 도달했는지 체크
        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("▷ 포크 최대 확장 완료! 수축 시퀀스로 전환합니다.")
            fork_state = "RETRACT"
            fork_target = 0.0  # 목표치를 최저점(0.0m)으로 변경
            
    elif fork_state == "RETRACT":
        # 완전히 들어왔는지(0.0m) 체크
        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("◁ 포크 완전 수축 완료! 다시 확장 시퀀스로 전환합니다.")
            fork_state = "EXTEND"
            fork_target = 0.7  # 목표치를 다시 최대 확장(0.7m)으로 변경

    # 주기적 로그 출력 (60프레임마다)
    if frame_count % 60 == 0:
        # print(f"Fork Target - Current: {fork_target - current_fork:.4f}m (State: {fork_state})")
        pass

    # 3. 매 스텝 설정된 목표 확장 명령을 공유 버퍼에 주입
    #lifting_amr.set_fork_extension(fork_target)

    # =========================================================================

    # 1. 현재 좌측 클로의 실제 각도 모니터링
    current_left_claw, _ = lifting_amr.get_claw_positions()
    
    # 2. 상태 머신 전환 조건 검사
    if claw_state == "GRIP":
        # 오므리기 목표 부근에 도달했는지 체크
        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("▶ 클로 움켜쥐기(GRIP) 완료! 열기 시퀀스로 전환합니다.")
            claw_state = "RELEASE"
            claw_target = -0.8  # 목표치를 반대 방향(열기)으로 변경
            
    elif claw_state == "RELEASE":
        # 열기 목표 부근에 도달했는지 체크
        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("◀ 클로 넓게 열기(RELEASE) 완료! 다시 움켜쥐기 시퀀스로 전환합니다.")
            claw_state = "GRIP"
            claw_target = 0.8  # 목표치를 다시 오므리기로 변경

    # 주기적 로그 출력 (60프레임마다)
    if frame_count % 60 == 0:
        # print(f"Claw Target - Current: {claw_target - current_left_claw:.4f}rad (State: {claw_state})")
        pass

    # 3. 매 스텝 설정된 목표 확장 명령을 공유 버퍼에 주입
    #lifting_amr.set_claw_grip(claw_target)

    # =========================================================================

    lifting_amr.apply_actuator_commands()
	
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)