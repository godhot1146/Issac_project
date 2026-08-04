import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController

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

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_asset = gym.load_asset(sim, asset_root, "urdf/tote/w/tote_h2.urdf", tote_opts)

# f_opts = gymapi.AssetOptions()
# f_opts.fix_base_link, f_opts.flip_visual_attachments = True, True
# f_opts.armature = 0.01
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
indy7_opts = gymapi.AssetOptions()
indy7_opts.fix_base_link = True
indy7_opts.density = 100.0
indy7_asset = gym.load_asset(sim, asset_root, "urdf/indy_description/urdf_files/indy7_v3_eye _1.urdf", indy7_opts)
indy7_handle = gym.create_actor(env, indy7_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.0)), "indy7_asset", -1, 0)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

frame_count = 0

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "move_x_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "move_x_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "move_y_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "move_y_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Z, "move_z_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "move_z_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "rot_roll_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "rot_roll_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "rot_pitch_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "rot_pitch_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "rot_yaw_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Y, "rot_yaw_neg")

#---------------------------------------------------------------------------------------
from isaacgym import gymtorch
import torch

# --- IK 검증용 셋업 ---

# 1. 관절 개수 및 구동 방식 설정 (position control로 설정해야 목표 각도로 실제 이동)
indy7_dof_count = gym.get_asset_dof_count(indy7_asset)
dof_props = gym.get_actor_dof_properties(env, indy7_handle)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"].fill(8000.0)
dof_props["damping"].fill(200.0)
gym.set_actor_dof_properties(env, indy7_handle, dof_props)

# 2. end-effector로 쓸 링크 이름 확인 (실제 urdf 안 마지막 링크명으로 바꿔야 함!)
#    모르면 아래 print로 전체 링크 목록 먼저 확인하세요.
body_names = gym.get_asset_rigid_body_names(indy7_asset)
print("=== indy7 링크 목록 ===")
print(body_names)
ee_name = body_names[-1]  # 우선 마지막 링크를 end-effector로 가정 (확인 후 수정 필요)

# 3. 시뮬레이션 텐서 API 활성화 (반드시 acquire 전에 호출)
gym.prepare_sim(sim)

# 4. 자코비안 텐서 획득
jacobian_tensor = gym.acquire_jacobian_tensor(sim, "indy7_asset")
jacobian = gymtorch.wrap_tensor(jacobian_tensor)

# 5. 강체(rigid body) 상태 텐서 획득 (end-effector 현재 위치 읽기용)
rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_state_tensor).view(1, -1, 13)  # env 1개 기준

# 6. DOF 상태 텐서 획득 (관절 각도 읽기/쓰기용)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_state_tensor).view(1, indy7_dof_count, 2)

# 7. end-effector의 actor 내부 인덱스 (jacobian/rb_states 슬라이싱용 — DOMAIN_ACTOR 기준이어야 함)
ee_index = gym.find_actor_rigid_body_index(env, indy7_handle, ee_name, gymapi.DOMAIN_ACTOR)
ee_index_actor = gym.find_actor_rigid_body_index(env, indy7_handle, ee_name, gymapi.DOMAIN_ACTOR)
ee_index_global = gym.find_actor_rigid_body_index(env, indy7_handle, ee_name, gymapi.DOMAIN_ENV)

# 8. IK 목표 좌표 (world 기준, 임의 지정 — 필요시 원하는 좌표로 수정)
target_pos = torch.tensor([-0.5, 0.0, 1.5], device=rb_states.device)

damping = 0.1

# 초기 목표 = 현재 tcp 위치/자세로 시작 (갑자기 튀지 않게)
gym.refresh_rigid_body_state_tensor(sim)
target_pos = rb_states[0, ee_index_global, 0:3].clone()
target_rot = rb_states[0, ee_index_global, 3:7].clone()  # 쿼터니언 (x,y,z,w)

move_speed = 0.01   # 프레임당 이동량 (m)
rot_speed = 0.1     # 프레임당 회전량 (rad)

print("rb_states:", rb_states.device)
print("dof_states:", dof_states.device)
print("jacobian:", jacobian.device)
print("jacobian shape:", jacobian.shape)   # (1, num_links-1, 6, num_dofs) 형태 예상
print("ee_index:", ee_index, "dof_count:", indy7_dof_count)
print("ee_index_actor:", ee_index_actor, "ee_index_global:", ee_index_global)

#---------------------------------------------------------------------------------------

env_origin = gym.get_env_origin(env)
print("env origin:", env_origin.x, env_origin.y, env_origin.z)
print("target_pos:", target_pos)

# =================================================================
# 직선 왕복 운동 파라미터
# =================================================================
base_z = 1.0  # indy7 스폰 시 베이스 z 높이 (create_actor에서 지정한 값)
mid_height = (base_z + target_pos[2].item()) / 2.0   # 베이스~현재 EE 높이의 중간

fixed_x = target_pos[0].item() + 0.3   # 현재 EE 위치에서 30cm 전방 (x축 기준, 필요시 y축으로 변경)
fixed_z = mid_height
y_center = target_pos[1].item()
y_amplitude = 0.5      # 좌우 ±50cm
osc_freq = 0.15         # 왕복 속도(Hz). 값을 키우면 더 빠르게 왕복

sim_time = 0.0

while not gym.query_viewer_has_closed(viewer):

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # --- 직선 왕복 운동 (좌우 사인파) ---
    sim_time += sim_params.dt
    y_offset = y_amplitude * np.sin(2 * np.pi * osc_freq * sim_time)
    target_pos = torch.tensor(
        [fixed_x, y_center + y_offset, fixed_z],
        device=rb_states.device,
        dtype=torch.float32   # ← 이 줄 추가
    )
	
    # --- 현재 상태 읽기 ---
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    ee_pos = rb_states[0, ee_index_global, 0:3]
    ee_rot = rb_states[0, ee_index_global, 3:7]  # (x,y,z,w)

    pos_err = target_pos - ee_pos

    # --- 회전 오차 계산 (쿼터니언 차이 → 축각 벡터) ---
    cur_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
    goal_q = gymapi.Quat(target_rot[0].item(), target_rot[1].item(), target_rot[2].item(), target_rot[3].item())
    err_q = goal_q * cur_q.inverse()
    if err_q.w < 0:  # 최단 경로 회전 보장
        err_q.x, err_q.y, err_q.z, err_q.w = -err_q.x, -err_q.y, -err_q.z, -err_q.w
    rot_err = torch.tensor([err_q.x, err_q.y, err_q.z], device=pos_err.device) * 2.0

    dpose = torch.cat([pos_err, rot_err]).unsqueeze(-1)

    j_eef = jacobian[0, ee_index_actor - 1, :, :indy7_dof_count]
    j_eef_T = j_eef.transpose(0, 1)
    lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
    d_theta = j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose

    cur_dof = dof_states[0, :, 0]
    target_dof_pos = cur_dof + d_theta.squeeze(-1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_dof_pos))

    origin = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
    sphere_geom = gymutil.WireframeSphereGeometry(
        0.05 * 1.4, 
        10, 
        10, 
        gymapi.Transform(p=origin), 
        (0.0, 0.0, 1.0)
    )
    gymutil.draw_lines(sphere_geom, gym, viewer, env, gymapi.Transform())
	
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)