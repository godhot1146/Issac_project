import os
import numpy as np
from isaacgym import gymapi, gymutil
from franka_controller import FrankaController
from carter_amr import CarterAMR
from low_amr import LowProfileAMR
from a_star_test import AStarPlanner

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

f_opts = gymapi.AssetOptions()
f_opts.fix_base_link, f_opts.flip_visual_attachments = True, True
f_opts.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", f_opts)

box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

shelf_opts = gymapi.AssetOptions()
shelf_opts.fix_base_link = False
shelf_opts.density = 100.0
cargo_shelf_asset = gym.load_asset(sim, asset_root, "urdf/cargo_shelf/cargo_shelf_test.urdf", shelf_opts)

c_opts = gymapi.AssetOptions()
c_opts.fix_base_link = False
carter_asset = gym.load_asset(sim, asset_root, "urdf/carter/carter.urdf", c_opts)

tray_opts = gymapi.AssetOptions()
tray_opts.fix_base_link, tray_opts.disable_gravity = False, True
tray_asset = gym.load_asset(sim, asset_root, "urdf/tray/traybox.urdf", tray_opts)

low_amr_opts = gymapi.AssetOptions()
low_amr_opts.fix_base_link = False
low_amr_asset = gym.load_asset(sim, asset_root, "urdf/low_amr/low_amr_edited.urdf", low_amr_opts)

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
# franka_handle = gym.create_actor(env, franka_asset, gymapi.Transform(), "franka", -1, 0)
# box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0.7, 0, 0.04)), "box", -1, 0)
# carter_handle = gym.create_actor(env, carter_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.7, 0.15)), "carter", -1, 0)
# tray_handle = gym.create_actor(env, tray_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.7, 0.25)), "tray", -1, 0)

low_amr_handle = gym.create_actor(env, low_amr_asset, gymapi.Transform(p=gymapi.Vec3(0.8, 0.77, 0.1)), "low_amr", -1, 0)
shelf_handle = gym.create_actor(env, cargo_shelf_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 0.77, 0.1)), "cargo_shelf", -1, 0)

# shelf_props = gym.get_actor_rigid_shape_properties(env, shelf_handle)
# for s in shelf_props: 
#     s.filter = 2  
#     s.friction = 15.0          # 일반 마찰력을 크게 상향 (기존 미지정)
#     s.rolling_friction = 5.0   # 구름 마찰력 추가
# # for s in shelf_props: 
# # 	s.filter = 2  
# gym.set_actor_rigid_shape_properties(env, shelf_handle, shelf_props)

low_amr_props = gym.get_actor_rigid_shape_properties(env, low_amr_handle)
for s in low_amr_props:
    s.friction = 15.0          # 10.0에서 15.0으로 상향
    s.rolling_friction = 5.0
# for s in low_amr_props:
# 	s.friction = 10.0
# 	s.rolling_friction = 5.0
gym.set_actor_rigid_shape_properties(env, low_amr_handle, low_amr_props)

# --- A* 플래너 및 컨트롤러 연동 ---
map_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../map")
npy_path = os.path.join(map_dir, "map.npy")
yaml_path = os.path.join(map_dir, "map_20260518_173220.yaml")

carter_planner = AStarPlanner(map_npy_path=npy_path, map_yaml_path=yaml_path, robot_radius=0.6)
# franka_ctrl = FrankaController(gym, env, franka_handle)
# amr_ctrl = CarterAMR(gym, env, carter_handle, tray_handle, planner=carter_planner)

low_planner = AStarPlanner(map_npy_path=npy_path, map_yaml_path=yaml_path, robot_radius=0.25)
low_amr_ctrl = LowProfileAMR(gym, env, low_amr_handle, planner=low_planner)

# 사용자 좌표 입력 수신
# try:
# 	input_x = float(input("바구니 AMR 목적지 X: "))
# 	input_y = float(input("바구니 AMR 목적지 Y: "))
# 	target_pos = np.array([input_x, input_y])
# except:
# 	target_pos = np.array([0.0, 0.0])

try:
	low_input_x = float(input("저상형 AMR 배달 목적지 X: "))
	low_input_y = float(input("저상형 AMR 배달 목적지 Y: "))
	low_delivery_coord = np.array([low_input_x, low_input_y])
except:
	low_delivery_coord = np.array([2.0, 2.0])  

# initial_c_states = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
# amr_ctrl.set_target((initial_c_states['pose']['p'][0]['x'], initial_c_states['pose']['p'][0]['y']), target_pos)

start_pos_coord = np.array([0.8, 0.77])  
target_pos_coord = np.array([3.0, 0.77]) 

low_amr_ctrl.set_lift("DOWN")
low_amr_ctrl.current_path = low_planner.plan_path(start_pos_coord, target_pos_coord, save_debug_img=True)
low_amr_ctrl.current_wp_idx = 0

low_amr_ctrl.viz_start_pos = start_pos_coord
low_amr_ctrl.viz_goal_pos = target_pos_coord

low_amr_state = "GO_TO_TARGET_COORD"
low_lift_timer = 0

# f_props = gym.get_actor_dof_properties(env, franka_handle)
# f_props["driveMode"].fill(gymapi.DOF_MODE_POS)
# f_props["stiffness"].fill(500.0); f_props["damping"].fill(10.0)
# gym.set_actor_dof_properties(env, franka_handle, f_props)

# c_props = gym.get_actor_dof_properties(env, carter_handle)
# c_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
# gym.set_actor_dof_properties(env, carter_handle, c_props)

# s_props = gym.get_actor_rigid_shape_properties(env, box_handle)
# for s in s_props: s.friction = 5.0
# gym.set_actor_rigid_shape_properties(env, box_handle, s_props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

is_loaded = False
frame_count = 0

while not gym.query_viewer_has_closed(viewer):
	# if franka_ctrl.update(frame_count):
	# 	if not is_loaded:
	# 		print("[태스크 동기화] Franka 매니퓰레이터 상자 적재 완료 ➔ 바구니 AMR 출발 승인")
	# 		is_loaded = True

	# curr_p, curr_r = amr_ctrl.sync_tray()
	# amr_ctrl.drive_to_target(curr_p, curr_r, is_loaded)
	
	low_p, low_r = low_amr_ctrl.get_current_pose()
	low_states = gym.get_actor_rigid_body_states(env, low_amr_handle, gymapi.STATE_ALL)
	low_curr_xy = np.array([low_states['pose']['p'][0]['x'], low_states['pose']['p'][0]['y']])
	
	if low_amr_state == "GO_TO_TARGET_COORD":
		low_amr_ctrl.is_docking_phase = True  
		low_amr_ctrl.set_lift("DOWN")
		dist_to_target = np.linalg.norm(low_curr_xy - target_pos_coord)
		if dist_to_target < 0.05: 
			low_amr_state = "LIFTING_AT_COORD"
			low_amr_ctrl.set_lift("UP")  
			low_lift_timer = 0
			print("\n[저상형 AMR FSM] 선반 중앙 정렬 완료 ➔ 슬로우 인양 시퀀스 개시")
			
	elif low_amr_state == "LIFTING_AT_COORD":
		low_amr_ctrl.set_lift("UP")
		low_lift_timer += 1
		if low_lift_timer > 120:  
			low_amr_ctrl.set_lift("UP")
			
			low_amr_ctrl.current_path = low_planner.plan_path(target_pos_coord, low_delivery_coord, save_debug_img=False)
			low_amr_ctrl.current_wp_idx = 0
			
			low_amr_state = "DELIVERY_TO_USER_COORD"
			print("\n=======================================================")
			print(f"[저상형 AMR FSM] 인양 완료 ➔ 자유 대각선 주행 개시")
			print(f" 배달 목적지   : [{low_delivery_coord[0]:.2f}, {low_delivery_coord[1]:.2f}]")
			print("=======================================================\n")
			
	elif low_amr_state == "DELIVERY_TO_USER_COORD":
		low_amr_ctrl.is_docking_phase = False  
		low_amr_ctrl.set_lift("UP")
		dist_to_delivery = np.linalg.norm(low_curr_xy - low_delivery_coord)
		
		if frame_count % 60 == 0:
			print(f"[실시간 트래킹] 로봇 위치 X/Y: [{low_curr_xy[0]:.2f}, {low_curr_xy[1]:.2f}] ➔ 남은거리: {dist_to_delivery:.2f}m")
			
		if dist_to_delivery < 0.05:
			low_amr_state = "MISSION_COMPLETE"
			low_amr_ctrl.current_path = []  
			print(f"\n[저상형 AMR FSM] 최종 목적지 좌표 {low_delivery_coord} 안전 파킹 완료!")
			
	elif low_amr_state == "MISSION_COMPLETE":
		# ★ 배송이 완료되면 짐을 내리도록 상태 반영
		low_amr_ctrl.set_lift("DOWN")
			
	low_amr_ctrl.drive_to_target(low_p, low_r)
	
	gym.simulate(sim)
	gym.fetch_results(sim, True)
	gym.step_graphics(sim)
	gym.draw_viewer(viewer, sim, True)
	
	text_color = gymapi.Vec3(1.0, 1.0, 1.0)
	
	gym.sync_frame_time(sim)
	frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)