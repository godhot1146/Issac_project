import os
import numpy as np
from isaacgym import gymapi, gymutil
from forklift_amr_controller_v1 import ForkliftAMR

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

box_size = 1.0
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False

# 💡 [튜닝 1] 박스 자체에 강력한 물리적 제동(Damping)을 걸어 외력에 의한 흔들림을 억제합니다.
# 값이 커질수록 가속도나 원심력에 의해 밀리는 현상이 사라집니다. (기존 0.2 -> 5.0~10.0 수준으로 버스트)
box_opts.linear_damping = 8.0
box_opts.angular_damping = 8.0

# 에셋 생성
box_asset = gym.create_box(sim, box_size, box_size, box_size, box_opts)

# 원하는 상자 무게(kg) 지정
target_box_mass = 7.0 

b_pose = gymapi.Transform(p=gymapi.Vec3(3.0, 0.0, 2.0))

# 환경(env)에 액터로 상자를 스폰합니다.
# b_handle = gym.create_actor(env, box_asset, b_pose, "cargo_box", -1, 0)



move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/pallet/v1/pallet_test.urdf", move_rack_opts)
pallet_handle = gym.create_actor(env, move_rack_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 0.0, 0.3)), "pallet_handle", -1, 0)

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False

forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)
forklift_handle = gym.create_actor(env, forklift_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.3)), "forklift_asset", -1, 1)

forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

waypoint_list = [
    (-2.0, -2.0),    # 1번 목적지
    (2.0, -2.0),    # 2번 목적지
    (2.0, 2.0),   # 3번 목적지
    (-2.0, 2.0),     # 복귀
	(0, 0)
]

waypoint_list = [
    (2.0, 0.0),    # 1번 목적지
    (0.0, 0.0)     # 복귀 목적지
]

# 💡 시퀀스 정밀 제어를 위한 단계 변수 세팅
# 0: 1번 목적지(2,0)로 주행 중
# 1: 목적지 도착 후 리프트 상승 중 및 대기
# 2: 리프트 유지한 채 복귀(0,0) 주행 중
sequence_stage = 0

current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 0.0          # 초기 상태에서는 리프트를 내린 상태(0.0m) 유지
LIFT_THRESHOLD = 0.01

CASTER_MIN_ANGLE = 0.0
CASTER_MAX_ANGLE = 0.523  

caster_target = 0.0

# 💡 캐스터 제어용 상태 변수 동일 구조 초기화
caster_state = "UP"
caster_target = 0.523  # 초기 목표 각도를 최고점(약 30도)으로 설정

# =========================================================================

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "START_SIM")

frame_count = 0
is_simulation_started = False  # 💡 스페이스바 입력 여부를 추적할 플래그 (기본값: 대기)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # -----------------------------------------------------------------
    # 💡 키보드 이벤트 처리 루틴 추가
    # -----------------------------------------------------------------
    for evt in gym.query_viewer_action_events(viewer):
        # 스페이스바가 눌린(action == "START_SIM") 순간을 포착
        if evt.action == "START_SIM" and evt.value > 0:
            if not is_simulation_started:
                is_simulation_started = True
                print("\n🚀 스페이스바 입력 감지! 로봇 주행 시퀀스를 시작합니다.")

    # 현재 실제 리프트 높이 피드백 수집
    current_height = forklift_amr.get_lift_height()

    # =========================================================================
    # 1. 시퀀스 단계별 주행 및 리프트 제어 락 (스페이스바 연동 + 도킹 후 1초 대기 추가)
    # =========================================================================
    
    # 💡 아직 스페이스바를 누르지 않았다면 속도 0m/s 및 리프트 0m로 제자리에 대기합니다.
    if not is_simulation_started:
        forklift_amr.set_twist_velocity(0.0, 0.0)
        lift_target = 0.0
        # 대기 시간 측정을 위한 변수 초기화
        wait_start_frame = None

    # [단계 0] 스페이스바가 눌린 이후, (2.0, 0.0) 목적지로 주행 단계
    elif sequence_stage == 0:
        lift_target = 0.0  # 갈 때는 리프트를 내리고 주행
        
        target_x, target_y = waypoint_list[0]
        is_arrived = forklift_amr.move_to_point(target_x, target_y)
        
        if is_arrived:
            print(f"★ [1번 목적지] ({target_x}, {target_y}) 도달 완료! 물리 안정을 위해 1초간 대기합니다...")
            forklift_amr.set_twist_velocity(0.0, 0.0)  # 정지
            wait_start_frame = frame_count             # 💡 1초 대기를 위한 기준 프레임 기록
            sequence_stage = 1                         # 대기 단계(단계 1)로 전환

    # [단계 1] 💡 [신규 추가] 목적지 도달 후 1초(60프레임) 동안 정적 대기하는 단계
    elif sequence_stage == 1:
        forklift_amr.set_twist_velocity(0.0, 0.0)      # 차체 완전 고정
        lift_target = 0.0                              # 대기 중에는 리프트 바닥 유지
        
        # 60프레임(1초)이 경과했는지 체크
        if (frame_count - wait_start_frame) >= 60:
            print(f"★ 1초 대기 완료. 리프트 상승을 시작합니다. (0.2m)")
            lift_target = 0.2                          # 리프트 목표치 20cm로 세팅
            sequence_stage = 2                         # 리프트 상승 단계(단계 2)로 전환

    # [단계 2] (2.0, 0.0)에서 리프트를 20cm(0.2m) 끝까지 올리는 단계
    elif sequence_stage == 2:
        forklift_amr.set_twist_velocity(0.0, 0.0)      # 리프트가 다 올라갈 때까지 차체 고정
        lift_target = 0.2                              # 목표 고도 지정
        
        # 리프트가 최고점(0.2m)에 안정적으로 도달했는지 체크
        if abs(lift_target - current_height) < LIFT_THRESHOLD:
            print(f"★ 리프트 20cm 상승 완료 및 고정! (0.0, 0.0)으로 복귀를 시작합니다.")
            sequence_stage = 3                         # 복귀 주행 단계(단계 3)로 전환

    # [단계 3] 리프트 높이(20cm)를 유지하면서 (0.0, 0.0)으로 복귀 주행 단계
    elif sequence_stage == 3:
        lift_target = 0.2  # 상승된 고도 그대로 유지 (락)
        
        target_x, target_y = waypoint_list[1]          # (0,0) 좌표 지정
        is_arrived = forklift_amr.move_to_point(target_x, target_y)
        
        if is_arrived:
            print(f"★ 최종 복귀 목적지 ({target_x}, {target_y}) 도달 완료! 복귀 시퀀스를 종료합니다.")
            forklift_amr.set_twist_velocity(0.0, 0.0)
            sequence_stage = 4                         # 모든 작업 완료 플래그

    # 모든 시퀀스 완료 후 최종 대기 상태
    else:
        forklift_amr.set_twist_velocity(0.0, 0.0)
        lift_target = 0.2  # 도착 후에도 리프트 높이 유지

    # 계산된 최종 리프트 타겟 높이 주입
    forklift_amr.set_lift_height(lift_target)

    # 1초(60프레임) 단위 모니터링 로그 출력
    if frame_count % 60 == 0:
        status_str = "대기 중 (스페이스바를 누르세요)" if not is_simulation_started else f"주행 중 (Stage: {sequence_stage})"
        print(f"[Run Log] Status: {status_str} | Lift Target: {lift_target}m | Current Height: {current_height:.4f}m")
    
    # -----------------------------------------------------------------
    # [명령 반영] 축적된 주행(속도)/리프트(위치) 명령 일괄 주입
    # -----------------------------------------------------------------
    forklift_amr.apply_actuator_commands()

    gym.sync_frame_time(sim)
    frame_count += 1

# while not gym.query_viewer_has_closed(viewer):
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
#     gym.clear_lines(viewer)

#     # -----------------------------------------------------------------
#     # 💡 키보드 이벤트 처리 루틴 추가
#     # -----------------------------------------------------------------
#     for evt in gym.query_viewer_action_events(viewer):
#         # 스페이스바가 눌린(action == "START_SIM") 순간을 포착
#         if evt.action == "START_SIM" and evt.value > 0:
#             if not is_simulation_started:
#                 is_simulation_started = True
#                 print("\n🚀 스페이스바 입력 감지! 로봇 주행 시퀀스를 시작합니다.")

#     # =========================================================================
#     # 직관적인 Twist 기반 속도 제어 명령 하달
#     # =========================================================================
	
#     # 현재 실제 리프트 높이 피드백 수집
#     current_height = forklift_amr.get_lift_height()

#     # =========================================================================
#     # 1. 시퀀스 단계별 주행 및 리프트 제어 락
#     # =========================================================================
    
#     # [단계 0] (2.0, 0.0) 목적지로 주행 단계
#     if sequence_stage == 0:
#         lift_target = 0.0  # 갈 때는 리프트를 내리고 주행
        
#         target_x, target_y = waypoint_list[0]
#         is_arrived = forklift_amr.move_to_point(target_x, target_y)
        
#         if is_arrived:
#             print(f"★ [1번 목적지] ({target_x}, {target_y}) 도달 완료! 리프트를 상승합니다.")
#             forklift_amr.set_twist_velocity(0.0, 0.0)  # 정지
#             lift_target = 0.2                          # 리프트 목표치 20cm로 세팅
#             sequence_stage = 1                         # 다음 단계로 전환

#     # [단계 1] (2.0, 0.0)에서 리프트를 20cm(0.2m) 끝까지 올리는 단계
#     elif sequence_stage == 1:
#         forklift_amr.set_twist_velocity(0.0, 0.0)      # 리프트가 다 올라갈 때까지 차체 고정
#         lift_target = 0.2                              # 목표 고도 지정
        
#         # 리프트가 최고점(0.2m)에 안정적으로 도달했는지 체크
#         if abs(lift_target - current_height) < LIFT_THRESHOLD:
#             print(f"★ 리프트 20cm 상승 완료 및 고정! (0.0, 0.0)으로 복귀를 시작합니다.")
#             sequence_stage = 2                         # 복귀 주행 단계로 전환

#     # [단계 2] 리프트 높이(20cm)를 유지하면서 (0.0, 0.0)으로 복귀 주행 단계
#     elif sequence_stage == 2:
#         lift_target = 0.2  # 💡 상승된 고도 그대로 유지 (락)
        
#         target_x, target_y = waypoint_list[1]          # (0,0) 좌표 지정
#         is_arrived = forklift_amr.move_to_point(target_x, target_y)
        
#         if is_arrived:
#             print(f"★ 최종 복귀 목적지 ({target_x}, {target_y}) 도달 완료! 복귀 시퀀스를 종료합니다.")
#             forklift_amr.set_twist_velocity(0.0, 0.0)
#             sequence_stage = 3                         # 모든 작업 완료 플래그

#     # 모든 시퀀스 완료 후 최종 대기 상태
#     else:
#         forklift_amr.set_twist_velocity(0.0, 0.0)
#         lift_target = 0.2  # 도착 후에도 리프트 높이 유지

#     # 계산된 최종 리프트 타겟 높이 주입
#     forklift_amr.set_lift_height(lift_target)

#     # 1초(60프레임) 단위 모니터링 로그 출력
#     if frame_count % 60 == 0:
#         print(f"[Run Log] Stage: {sequence_stage} | Lift Target: {lift_target}m | Current Height: {current_height:.4f}m")
    
#     # -----------------------------------------------------------------
#     # [주행 제어] 직교 분리형 그리드 이동 명령 하달
#     # -----------------------------------------------------------------
#     # if current_wp_idx < len(waypoint_list):
#     #     target_x, target_y = waypoint_list[current_wp_idx]
        
#     #     # 직교 자율주행 실행 후 도달 시 True 반환
#     #     is_arrived = forklift_amr.move_to_point(target_x, target_y)
        
#     #     if is_arrived:
#     #         print(f"★ [{current_wp_idx + 1}번 목적지] ({target_x}, {target_y}) 도달 완료! 다음 좌표로 이동합니다.")
#     #         current_wp_idx += 1
#     # else:
#     #     # 모든 경유지 순회 완료 시 대기
#     #     forklift_amr.set_twist_velocity(0.0, 0.0)
        
#     # -----------------------------------------------------------------

#     # current_height = forklift_amr.get_lift_height()
    
#     # if lift_state == "UP":
#     #     # 최고점(0.2m) 근처에 도달했는지 체크
#     #     if abs(lift_target - current_height) < LIFT_THRESHOLD:
#     #         lift_state = "DOWN"
#     #         lift_target = 0.0  # 목표치를 최저점(0.2m)으로 변환
            
#     # elif lift_state == "DOWN":
#     #     # 최저점(0.0m) 근처에 도달했는지 체크
#     #     if abs(lift_target - current_height) < LIFT_THRESHOLD:
#     #         lift_state = "UP"
#     #         lift_target = 0.2  # 목표치를 다시 최고점(0.2m)으로 일치화

#     # # 1초(60프레임) 단위 모니터링 로그 출력
#     # if frame_count % 60 == 0:
#     #     # print(f"[Lift Log] Target: {lift_target}m | Current: {current_height:.4f}m | Error: {lift_target - current_height:.4f}m")
#     #     pass
    
#     # forklift_amr.set_lift_height(lift_target)

#     # =========================================================================
#     # 2. 캐스터 레벨링 제어 로직 (리프트 스타일 독립 스위칭 구조) 💡
#     # =========================================================================
#     # current_caster_angle = forklift_amr.get_caster_angle()

#     # CASTER_THRESHOLD = 0.01  # 각도 도달 인정 공차 (약 0.5도)

#     # # 초기 상태 변수가 메인 상단에 정의되어 있어야 합니다 (기본값 예시: caster_state = "UP", caster_target = 0.523)
#     # if caster_state == "UP":
#     #     # 목표 각도(최고점 0.523 rad) 근처에 도달했는지 체크
#     #     if abs(caster_target - current_caster_angle) < CASTER_THRESHOLD:
#     #         caster_state = "DOWN"
#     #         caster_target = 0.0  # 최고점에 도달했으므로 다음 목표치를 최저점(0.0 rad)으로 변환
            
#     # elif caster_state == "DOWN":
#     #     # 목표 각도(최저점 0.0 rad) 근처에 도달했는지 체크
#     #     if abs(caster_target - current_caster_angle) < CASTER_THRESHOLD:
#     #         caster_state = "UP"
#     #         caster_target = 0.523  # 최저점에 도달했으므로 다음 목표치를 다시 최고점(0.523 rad)으로 일치화

#     # # 1초(60프레임) 단위 캐스터 상태 모니터링 로그 출력 (필요 시 주석 해제)
#     # if frame_count % 60 == 0:
#     #     # print(f"[Caster Log] State: {caster_state} | Target: {caster_target:.3f}rad | Current: {current_caster_angle:.4f}rad")
#     #     pass
        
#     # 캐스터 단독 명령 지정
#     # forklift_amr.set_caster_angle(caster_target)
		
#     # -----------------------------------------------------------------
#     # [명령 반영] 축적된 주행(속도)/리프트(위치) 명령 일괄 주입
#     # -----------------------------------------------------------------
#     forklift_amr.apply_actuator_commands()

#     gym.sync_frame_time(sim)
#     frame_count += 1

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)