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

sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001
sim_params.physx.bounce_threshold_velocity = 0.2

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# ★ 은은하고 부드러운 스튜디오 조명 유지 (그림자 최소화)
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0.2, 0.3, -1.0))

# 2. 바닥 평면 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

# =================================================================
# 3. 에셋 규격 및 가상 맵 정의
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# ★ 공장 사이즈 변경 (6.75m x 12.0m)
room_width = 6.75 # -> generate_A*_map
room_length = 12.0 # -> generate_A*_map
wall_height = 10.0
wall_thickness = 0.2 # -> generate_A*_map

pillar_x_size = 0.4  
pillar_y_size = 0.4

# static asset 바닥 및 외벽 직사각형 비율 적용
floor_asset = gym.create_box(sim, room_width, room_length, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_length, wall_height, env_opts)
wall_y = gym.create_box(sim, room_width, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, pillar_x_size, pillar_y_size, wall_height, env_opts)

shelf_opts = gymapi.AssetOptions()
shelf_opts.fix_base_link = False
shelf_opts.density = 100.0
cargo_shelf_asset = gym.load_asset(sim, asset_root, "urdf/cargo_shelf/cargo_shelf.urdf", shelf_opts)

# dynamic asset
f_opts = gymapi.AssetOptions()
f_opts.fix_base_link, f_opts.flip_visual_attachments = True, True
f_opts.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", f_opts)

box_size = 0.22
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = True
box_asset = gym.create_box(sim, box_size, box_size, box_size, box_opts)

c_opts = gymapi.AssetOptions()
c_opts.fix_base_link = False
carter_asset = gym.load_asset(sim, asset_root, "urdf/carter/carter.urdf", c_opts)

tray_opts = gymapi.AssetOptions()
tray_opts.fix_base_link, tray_opts.disable_gravity = False, True
tray_asset = gym.load_asset(sim, asset_root, "urdf/tray/traybox.urdf", tray_opts)

low_amr_opts = gymapi.AssetOptions()
low_amr_opts.fix_base_link = False
low_amr_asset = gym.load_asset(sim, asset_root, "urdf/low_amr/low_amr.urdf", low_amr_opts)

vacuum_gripper_opts = gymapi.AssetOptions()
vacuum_gripper_opts.fix_base_link, vacuum_gripper_opts.flip_visual_attachments = True, True
vacuum_gripper_opts.armature = 0.01
vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper.urdf", vacuum_gripper_opts)


# =================================================================
# 4. 직사각형 환경 생성 및 외벽/기둥 배치
# =================================================================
env = gym.create_env(sim, gymapi.Vec3(-room_width/2, -room_length/2, 0), gymapi.Vec3(room_width/2, room_length/2, wall_height), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# ★ 가로/세로 절반 길이 계산
half_w = room_width / 2
half_l = room_length / 2

w_back = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_w, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_w, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_l, wall_height/2)), "wall_right", 0, 0)
w_left = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_l, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
	gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

p_offset_w = half_w - 0.25
p_offset_l = half_l - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset_w, p_offset_w]:
	for y in [-p_offset_l, p_offset_l]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

pillar_side_positions = np.arange(-2.5, 3.5, 2.5)
for y_pos in pillar_side_positions:
	for x_pos in [-p_offset_w, p_offset_w]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)),f"side_pillar_{x_pos}_{y_pos}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# =================================================================
# 🏭 [레이아웃 재설계] 양쪽 벽면 밀착 정렬 & 중앙 로봇 고속도로 확보
# =================================================================
# 좌측(X=-3.5)과 우측(X=3.5) 벽을 따라 랙을 6개씩 일렬로 배치합니다.
# 저상형 AMR의 타겟 좌표인 [3.5, 0.77] 위치에는 랙이 정확히 위치하도록 좌표를 맞췄습니다.
shelf_positions = [
	# 좌측 벽면 (X=-2.3)
	[-2.3, 3.49], [-2.3, 2.13], [-2.3, 0.77], [-2.3, -0.59],
	# 뒤쪽 벽면 (Y=5.0)
	[-2.3, 5.0], [-0.5, 5.0], [1.0, 5.0], [2.5, 5.0],
	# 우측 벽면 (X=2.3)
	[2.3, 3.49], [2.3, 2.13], [2.3, 0.77], [2.3, -0.59], [2.3, -1.95], [2.3, -3.31], [2.3, -4.76]
]

cardboard_color = gymapi.Vec3(0.76, 0.60, 0.42)

for idx, pos in enumerate(shelf_positions):
	pose = gymapi.Transform(p=gymapi.Vec3(pos[0], pos[1], 0.06))
	s_handle = gym.create_actor(env, cargo_shelf_asset, pose, f"shelf_{idx}", -1, 0)

	s_props = gym.get_actor_rigid_shape_properties(env, s_handle)
	for s in s_props:
		s.filter = 2
		s.friction = 5.0
	gym.set_actor_rigid_shape_properties(env, s_handle, s_props)

	# 5개의 상단 선반 각각에 4x4 그리드(16개) 박스를 빈틈없이 적재 (총 960여 개)
	for z_level in [1.15, 2.05, 2.95, 3.85, 4.75]:
		for dx in [-0.38, -0.13, 0.13, 0.38]:
			for dy in [-0.38, -0.13, 0.13, 0.38]:
				rand_x = dx + np.random.uniform(-0.01, 0.01)
				rand_y = dy + np.random.uniform(-0.01, 0.01)
				b_pose = gymapi.Transform(p=gymapi.Vec3(pos[0] + rand_x, pos[1] + rand_y, z_level + 0.15))

				b_handle = gym.create_actor(env, box_asset, b_pose, f"cargo_box_{idx}_{z_level}_{dx}_{dy}", -1, 0)
				gym.set_rigid_body_color(env, b_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cardboard_color)

				b_props = gym.get_actor_rigid_shape_properties(env, b_handle)
				for b in b_props: b.friction = 3.0
				gym.set_actor_rigid_shape_properties(env, b_handle, b_props)

# =================================================================
# 🛠️ 로봇 인스턴스 스폰 (중앙 하이웨이 배치)
# =================================================================
vacuum_gripper_handle = gym.create_actor(env, vacuum_gripper_asset, gymapi.Transform(p=gymapi.Vec3(-1.0, 0.0, 0.055)), "vacuum_gripper_asset", -1, 0)
carter_handle = gym.create_actor(env, carter_asset, gymapi.Transform(p=gymapi.Vec3(0, 2.7, 0.30)), "carter", -1, 0)
tray_handle = gym.create_actor(env, tray_asset, gymapi.Transform(p=gymapi.Vec3(0, 2.7, 0.50)), "tray", -1, 0)

low_amr_handle = gym.create_actor(env, low_amr_asset, gymapi.Transform(p=gymapi.Vec3(0.5, 0.77, 0.20)), "low_amr", -1, 0)

low_amr_props = gym.get_actor_rigid_shape_properties(env, low_amr_handle)
for s in low_amr_props:
	s.friction = 10.0
	s.rolling_friction = 5.0
gym.set_actor_rigid_shape_properties(env, low_amr_handle, low_amr_props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# 바뀐 레이아웃(중앙 고속도로)이 한눈에 보이도록 카메라를 정면 끝단에 배치
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-3.5, -4.5, 12.0), gymapi.Vec3(0.0, 0.0, 0.0))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):
	gym.simulate(sim)
	gym.fetch_results(sim, True)
	gym.step_graphics(sim)
	gym.draw_viewer(viewer, sim, True)
	
	gym.sync_frame_time(sim)
	frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)