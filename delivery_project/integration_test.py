import os
import numpy as np
from isaacgym import gymapi, gymutil

# 1. 시뮬레이션 및 물리 엔진 초기화
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Franka-AMR Collaborative Delivery")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12 

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

gym.set_light_parameters(sim, 0, gymapi.Vec3(0.95, 0.95, 0.95), gymapi.Vec3(0.3, 0.3, 0.35), gymapi.Vec3(1.0, -1.0, -1.0))

# 2. 바닥 평면 생성 (물리 연산용 무한 평면)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05 
gym.add_ground(sim, plane_params)

# 3. 에셋 로드
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/isaac_assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True
room_size = 16.0
wall_height = 5.0
wall_thickness = 0.2

# 바닥 에셋 (두께 0.05의 박스 형태)
floor_asset = gym.create_box(sim, room_size, room_size, 0.05, env_opts)
wall_x_asset = gym.create_box(sim, wall_thickness, room_size, wall_height, env_opts)
wall_y_asset = gym.create_box(sim, room_size, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.5, 0.5, wall_height, env_opts)

f_opts = gymapi.AssetOptions()
f_opts.fix_base_link = True
f_opts.flip_visual_attachments = True
f_opts.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", f_opts)

box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

c_opts = gymapi.AssetOptions()
c_opts.fix_base_link = False
carter_asset = gym.load_asset(sim, asset_root, "urdf/carter/carter.urdf", c_opts)

tray_opts = gymapi.AssetOptions()
tray_opts.fix_base_link = False
# 동기화할 때 중력 때문에 떨리는 것을 막기 위해 Tray 중력 비활성화
tray_opts.disable_gravity = True
tray_asset = gym.load_asset(sim, asset_root, "urdf/tray/traybox.urdf", tray_opts)

struct_opts = gymapi.AssetOptions()
struct_opts.fix_base_link = True
struct_opts.use_mesh_materials = True
cabinet_asset = gym.load_asset(sim, asset_root, "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf", struct_opts)

# 4. 환경 및 액터 생성
env = gym.create_env(sim, gymapi.Vec3(-room_size/2, -room_size/2, 0), gymapi.Vec3(room_size/2, room_size/2, room_size), 1)

warehouse_gray = gymapi.Vec3(0.5, 0.5, 0.5)
# 두께 0.05의 바닥을 z=0.03에 배치하여 윗면이 z=0에 오도록 함
floor_handle = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, warehouse_gray)

# 외곽 벽 배치
w_back = gym.create_actor(env, wall_x_asset, gymapi.Transform(p=gymapi.Vec3(-room_size/2, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x_asset, gymapi.Transform(p=gymapi.Vec3(room_size/2, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y_asset, gymapi.Transform(p=gymapi.Vec3(0, -room_size/2, wall_height/2)), "wall_right", 0, 0)
w_left = gym.create_actor(env, wall_y_asset, gymapi.Transform(p=gymapi.Vec3(0, room_size/2, wall_height/2)), "wall_left", 0, 0)

# 벽 색상은 진회색으로 통일
wall_gray = gymapi.Vec3(0.3, 0.3, 0.3)
for w in [w_back, w_front, w_right, w_left]:
	gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

# [물류 창고 레이아웃: 하이 랙 시스템]
stack_height, cabinet_scale = 3, 2.0
cabinet_h = 0.7
safety_orange = gymapi.Vec3(0.9, 0.4, 0.1)
x_positions = [-7.1, -5.4, -3.7, -2.0, -0.3, 1.4, 3.1, 4.8, 6.5]
x_positions2 = [-3.7, -2.0, -0.3, 1.4, 3.1, 4.8, 6.5]

for x in x_positions:
	for level in range(stack_height):
		z_pos = level * (cabinet_h * cabinet_scale) + 1
		c_pose_back = gymapi.Transform(p=gymapi.Vec3(x, 7.0, z_pos), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi/2))
		c_handle_b = gym.create_actor(env, cabinet_asset, c_pose_back, f"rack_b_{x}_{level}", 0, 0)
		gym.set_actor_scale(env, c_handle_b, cabinet_scale)
		gym.set_rigid_body_color(env, c_handle_b, 0, gymapi.MESH_VISUAL_AND_COLLISION, safety_orange)
		
		c_pose_front = gymapi.Transform(p=gymapi.Vec3(x, -7.0, z_pos), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi/2))
		c_handle_f = gym.create_actor(env, cabinet_asset, c_pose_front, f"rack_f_{x}_{level}", 0, 0)
		gym.set_actor_scale(env, c_handle_f, cabinet_scale)
		gym.set_rigid_body_color(env, c_handle_f, 0, gymapi.MESH_VISUAL_AND_COLLISION, safety_orange)
		
for x in x_positions2:
	for level in range(stack_height):
		z_pos = level * (cabinet_h * cabinet_scale) + 1
		c_pose_back = gymapi.Transform(p=gymapi.Vec3(x, 4.0, z_pos), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi/2))
		c_handle_b = gym.create_actor(env, cabinet_asset, c_pose_back, f"rack_b_{x}_{level}", 0, 0)
		gym.set_actor_scale(env, c_handle_b, cabinet_scale)
		gym.set_rigid_body_color(env, c_handle_b, 0, gymapi.MESH_VISUAL_AND_COLLISION, safety_orange)
		
		c_pose_front = gymapi.Transform(p=gymapi.Vec3(x, -4.0, z_pos), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi/2))
		c_handle_f = gym.create_actor(env, cabinet_asset, c_pose_front, f"rack_f_{x}_{level}", 0, 0)
		gym.set_actor_scale(env, c_handle_f, cabinet_scale)
		gym.set_rigid_body_color(env, c_handle_f, 0, gymapi.MESH_VISUAL_AND_COLLISION, safety_orange)

# 기둥 배치
p_offset = room_size/2 - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

# 네 모서리 기둥 (w_back, w_front 양 끝)
for x in [-p_offset, p_offset]:
	for y in [-p_offset, p_offset]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)
# 2. 좌우 벽면(w_right, w_left)을 따라 2m 간격으로 기둥 배치
pillar_side_positions = np.arange(-6.0, 7.0, 3.0)

for y_pos in pillar_side_positions:
	for x_pos in [-p_offset, p_offset]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)),f"side_pillar_{x_pos}_{y_pos}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# 로봇 및 상자 배치
franka_handle = gym.create_actor(env, franka_asset, gymapi.Transform(), "franka", -1, 0)
box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0.7, 0, 0.04)), "box", -1, 0)
carter_handle = gym.create_actor(env, carter_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.7, 0.1)), "carter", -1, 0)
tray_handle = gym.create_actor(env, tray_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.7, 0.25)), "tray", -1, 0)

gym.set_rigid_body_color(env, carter_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))
gym.set_rigid_body_color(env, tray_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.3, 0.3, 0.3))

# 5. 제어 및 마찰력 설정
f_props = gym.get_actor_dof_properties(env, franka_handle)
f_props["driveMode"].fill(gymapi.DOF_MODE_POS)
f_props["stiffness"].fill(800.0)
f_props["damping"].fill(40.0)
gym.set_actor_dof_properties(env, franka_handle, f_props)

c_props = gym.get_actor_dof_properties(env, carter_handle)
c_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
c_props["damping"].fill(5.0)
gym.set_actor_dof_properties(env, carter_handle, c_props)

s_props = gym.get_actor_rigid_shape_properties(env, box_handle)
for s in s_props: s.friction = 5.0
gym.set_actor_rigid_shape_properties(env, box_handle, s_props)

# 6. 동작 시퀀스 정의
ready_A = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
pick_A = [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.04, 0.04] 
close_A = [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.0, 0.0]    
lift_A = [0, -0.5, 0, -2.0, 0, 1.50, 0.90, 0.0, 0.0]      
rotate_B = [1.571, -0.5, 0, -2.0, 0, 2.50, 0.90, 0.0, 0.0]
place_B = [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6, 0.0, 0.0] 
release_B = [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6, 0.04, 0.04]
lift_up_B = [1.571, -0.5, 0, -1.45, 0, 1.50, 0.6, 0.04, 0.04] 

print("-" * 30)
try:
    input_x = float(input("AMR 배달 목적지 X: "))
    input_y = float(input("AMR 배달 목적지 Y: "))
    target_pos = np.array([input_x, input_y])
except ValueError:
    target_pos = np.array([3.0, 4.0])

is_loaded = False
franka_done = False
kp_linear = 1.5
kp_angular = 15.0
min_dist = 0.25

# 7. 실행 루프
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(4.0, 2.0, 5.0), gymapi.Vec3(0, 0.5, 0.5))

frame_count = 0
while not gym.query_viewer_has_closed(viewer):
    if not franka_done:
        step = (frame_count // 100)
        if step == 0: targets = ready_A
        elif step == 1: targets = pick_A
        elif step == 2: targets = close_A
        elif step == 3: targets = lift_A
        elif step == 4: targets = rotate_B
        elif step == 5: targets = place_B
        elif step == 6: targets = release_B
        elif step == 7: 
            targets = lift_up_B
            is_loaded = True
        else:
            targets = ready_A
            franka_done = True
        gym.set_actor_dof_position_targets(env, franka_handle, np.array(targets, dtype=np.float32))

    # AMR 주행 및 Tray 위치 실시간 강제 동기화
    c_states = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
    curr_p, curr_r = c_states['pose']['p'][0], c_states['pose']['r'][0]
    
    t_states = gym.get_actor_rigid_body_states(env, tray_handle, gymapi.STATE_ALL)
    t_states['pose']['p'][0]['x'] = curr_p['x']
    t_states['pose']['p'][0]['y'] = curr_p['y']
    t_states['pose']['p'][0]['z'] = curr_p['z'] + 0.07 # 섀시 위 7cm 지점에 딱 붙임
    t_states['pose']['r'][0] = curr_r
    gym.set_actor_rigid_body_states(env, tray_handle, t_states, gymapi.STATE_ALL)

    if is_loaded:
        c_states = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
        curr_p = c_states['pose']['p'][0]
        curr_r = c_states['pose']['r'][0]
        curr_yaw = np.arctan2(2.0 * (curr_r['w'] * curr_r['z'] + curr_r['x'] * curr_r['y']), 
                              1.0 - 2.0 * (curr_r['y']**2 + curr_r['z']**2))

        err_x, err_y = target_pos[0] - curr_p['x'], target_pos[1] - curr_p['y']
        dist_error = np.sqrt(err_x**2 + err_y**2)
        target_yaw = np.arctan2(err_y, err_x)
        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        if dist_error < min_dist:
            v_lin, v_ang = 0.0, 0.0
        else:
            v_lin = np.clip(kp_linear * dist_error * np.cos(yaw_error), -2.0, 2.0)
            v_ang = np.clip(kp_angular * yaw_error, -4.0, 4.0)
    else:
        v_lin, v_ang = 0.0, 0.0

    ls, rs = v_lin - v_ang, v_lin + v_ang
    gym.set_actor_dof_velocity_targets(env, carter_handle, np.array([ls, rs, ls, rs], dtype=np.float32))

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
