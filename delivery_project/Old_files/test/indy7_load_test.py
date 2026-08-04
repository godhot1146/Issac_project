import os
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v2 import FrankaController

class LinearMotionGenerator:
    """직선 경로를 따라 목표를 매 프레임 조금씩 이동시켜주는 궤적 생성기"""
    def __init__(self, linear_speed=0.15, angular_speed=1.0, dt=1.0/60.0):
        self.linear_speed = linear_speed    # m/s
        self.angular_speed = angular_speed  # rad/s
        self.dt = dt
        self.active = False

    def start(self, start_pos, start_quat, goal_pos, goal_quat):
        """새 목표가 들어오면 즉시 호출 — 이 프레임부터 바로 움직이기 시작"""
        self.start_pos = start_pos.clone()
        self.goal_pos = goal_pos.clone()
        self.start_quat = gymapi.Quat(*start_quat.tolist())
        self.goal_quat = gymapi.Quat(*goal_quat.tolist())

        dist = (goal_pos - start_pos).norm().item()
        ang = self._quat_angle(self.start_quat, self.goal_quat)

        # 위치/회전 중 더 오래 걸리는 쪽 기준으로 전체 이동시간 산정
        t_lin = dist / self.linear_speed if self.linear_speed > 0 else 0
        t_ang = ang / self.angular_speed if self.angular_speed > 0 else 0
        self.duration = max(t_lin, t_ang, 1e-6)
        self.elapsed = 0.0
        self.active = True

    def _quat_angle(self, q1, q2):
        dot = abs(q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w)
        dot = min(1.0, dot)
        return 2 * np.arccos(dot)

    def step(self):
        """매 프레임 호출 — 지금 이 순간 IK가 따라가야 할 waypoint 반환"""
        if not self.active:
            return None, None

        self.elapsed += self.dt
        alpha = min(1.0, self.elapsed / self.duration)  # 0~1 진행률

        # 위치: 선형보간(LERP) → 직선 경로 자체가 수학적으로 보장됨
        way_pos = self.start_pos + (self.goal_pos - self.start_pos) * alpha

        # 회전: 구면선형보간(SLERP)
        way_quat = slerp_quat(self.start_quat, self.goal_quat, alpha)
        way_quat_t = torch.tensor([way_quat.x, way_quat.y, way_quat.z, way_quat.w],
                                    device=self.start_pos.device)

        if alpha >= 1.0:
            self.active = False

        return way_pos, way_quat_t

def slerp_quat(q1, q2, alpha):
    """gymapi.Quat 두 개를 구면선형보간. alpha=0이면 q1, alpha=1이면 q2 반환."""
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w

    dot = x1*x2 + y1*y2 + z1*z2 + w1*w2

    # 최단 경로 보장: dot이 음수면 한쪽 쿼터니언 부호를 뒤집음
    if dot < 0.0:
        x2, y2, z2, w2 = -x2, -y2, -z2, -w2
        dot = -dot

    dot = min(1.0, max(-1.0, dot))  # 수치 오차로 인한 범위 이탈 방지

    if dot > 0.9995:
        # 각도가 매우 작으면 선형보간으로 근사 (0으로 나누기 방지)
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        z = z1 + alpha * (z2 - z1)
        w = w1 + alpha * (w2 - w1)
        result = gymapi.Quat(x, y, z, w)
    else:
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        x = s1*x1 + s2*x2
        y = s1*y1 + s2*y2
        z = s1*z1 + s2*z2
        w = s1*w1 + s2*w2
        result = gymapi.Quat(x, y, z, w)

    # 정규화 (수치 오차 누적 방지)
    norm = np.sqrt(result.x**2 + result.y**2 + result.z**2 + result.w**2)
    return gymapi.Quat(result.x/norm, result.y/norm, result.z/norm, result.w/norm)

def scale_to_joint_limits(d_theta, dt, max_joint_vel):
    """
    d_theta: 이번 스텝에 필요한 관절 변화량 (rad)
    dt: 시간 간격
    max_joint_vel: 관절별 최대속도 (rad/s), shape=(6,)
    
    반환: 관절 간 비율(=직선 방향)을 유지한 채 균일하게 축소된 d_theta
    """
    required_vel = torch.abs(d_theta.squeeze(-1)) / dt          # 이번 스텝에 요구되는 속도
    ratio = required_vel / max_joint_vel                         # 한계 대비 얼마나 초과하는지
    max_ratio = torch.max(ratio)

    if max_ratio > 1.0:
        scale = 1.0 / max_ratio          # 가장 심하게 초과하는 관절 기준으로 전체를 같은 비율로 축소
        return d_theta * scale
    return d_theta

def weighted_dls(j_eef, dpose, weights, damping):
    """
    weights: 6차원 텐서 [wx, wy, wz, wroll, wpitch, wyaw]
             값이 클수록 "그 축은 강하게 고정" / 값이 작을수록 "그 축은 자유롭게 이동 허용"
    """
    W = torch.diag(weights)
    JT_W = j_eef.transpose(0, 1) @ W
    lmbda = torch.eye(j_eef.shape[1], device=j_eef.device) * (damping ** 2)
    d_theta = torch.inverse(JT_W @ j_eef + lmbda) @ (JT_W @ dpose)
    return d_theta

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

#indy7_asset = gym.load_asset(sim, asset_root, "urdf/indy_description/urdf_files/indy7_v3_eye _1.urdf", indy7_opts)
indy7_asset = gym.load_asset(sim, asset_root, "urdf/indy_description/urdf_files/indy7_v3_eye _vacuum.urdf", indy7_opts)

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
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "rot_roll_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Y, "rot_roll_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_U, "rot_pitch_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_I, "rot_pitch_neg")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "rot_yaw_pos")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "rot_yaw_neg")

#---------------------------------------------------------------------------------------
from isaacgym import gymtorch
import torch

# --- IK 검증용 셋업 ---

# 1. 관절 개수 및 구동 방식 설정 (position control로 설정해야 목표 각도로 실제 이동)
indy7_dof_count = gym.get_asset_dof_count(indy7_asset)
dof_props = gym.get_actor_dof_properties(env, indy7_handle)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"].fill(20000.0)
dof_props["damping"].fill(1000.0)
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
rot_speed = 0.15     # 프레임당 회전량 (rad)

print("rb_states:", rb_states.device)
print("dof_states:", dof_states.device)
print("jacobian:", jacobian.device)
print("jacobian shape:", jacobian.shape)   # (1, num_links-1, 6, num_dofs) 형태 예상
print("ee_index:", ee_index, "dof_count:", indy7_dof_count)
print("ee_index_actor:", ee_index_actor, "ee_index_global:", ee_index_global)

#---------------------------------------------------------------------------------------

# 축 인덱스: 0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw
AXIS_MAP = {
    "move_x_pos": 0, "move_x_neg": 0,
    "move_y_pos": 1, "move_y_neg": 1,
    "move_z_pos": 2, "move_z_neg": 2,
    "rot_roll_pos": 3, "rot_roll_neg": 3,
    "rot_pitch_pos": 4, "rot_pitch_neg": 4,
    "rot_yaw_pos": 5, "rot_yaw_neg": 5,
}

LOCK_WEIGHT = 200.0   # 고정할 축 가중치 (클수록 강하게 붙잡음)
FREE_WEIGHT = 1.0     # 움직일 축 가중치

active_axis_weights = torch.full((6,), LOCK_WEIGHT)  # 초기: 전부 고정 (정지 상태 유지)

motion_gen = LinearMotionGenerator(linear_speed=5.00, angular_speed=1.0, dt=1.0/60.0)

env_origin = gym.get_env_origin(env)
print("env origin:", env_origin.x, env_origin.y, env_origin.z)
print("target_pos:", target_pos)

while not gym.query_viewer_has_closed(viewer):

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # # --- 키 입력으로 목표 갱신 ---
    # for evt in gym.query_viewer_action_events(viewer):
    #     if evt.value > 0:  # 키 눌림
    #         if evt.action == "move_x_pos": target_pos[0] += move_speed
    #         elif evt.action == "move_x_neg": target_pos[0] -= move_speed
    #         elif evt.action == "move_y_pos": target_pos[1] += move_speed
    #         elif evt.action == "move_y_neg": target_pos[1] -= move_speed
    #         elif evt.action == "move_z_pos": target_pos[2] += move_speed
    #         elif evt.action == "move_z_neg": target_pos[2] -= move_speed
    #         elif evt.action in ("rot_roll_pos","rot_roll_neg","rot_pitch_pos","rot_pitch_neg","rot_yaw_pos","rot_yaw_neg"):
    #             axis = {"rot_roll_pos":[1,0,0],"rot_roll_neg":[-1,0,0],
    #                      "rot_pitch_pos":[0,1,0],"rot_pitch_neg":[0,-1,0],
    #                      "rot_yaw_pos":[0,0,1],"rot_yaw_neg":[0,0,-1]}[evt.action]
    #             dq = gymapi.Quat.from_axis_angle(gymapi.Vec3(*axis), rot_speed)
    #             cur_q = gymapi.Quat(*target_rot.tolist())
    #             new_q = dq * cur_q
    #             target_rot = torch.tensor([new_q.x, new_q.y, new_q.z, new_q.w], device=target_rot.device)
    # --- 키 입력으로 목표 갱신 ---
    goal_changed = False
    pressed_axes = set()

    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            if evt.action == "move_x_pos": target_pos[0] += move_speed; goal_changed = True
            elif evt.action == "move_x_neg": target_pos[0] -= move_speed; goal_changed = True
            elif evt.action == "move_y_pos": target_pos[1] += move_speed; goal_changed = True
            elif evt.action == "move_y_neg": target_pos[1] -= move_speed; goal_changed = True
            elif evt.action == "move_z_pos": target_pos[2] += move_speed; goal_changed = True
            elif evt.action == "move_z_neg": target_pos[2] -= move_speed; goal_changed = True
            elif evt.action in ("rot_roll_pos","rot_roll_neg","rot_pitch_pos","rot_pitch_neg","rot_yaw_pos","rot_yaw_neg"):
                axis = {"rot_roll_pos":[1,0,0],"rot_roll_neg":[-1,0,0],
                         "rot_pitch_pos":[0,1,0],"rot_pitch_neg":[0,-1,0],
                         "rot_yaw_pos":[0,0,1],"rot_yaw_neg":[0,0,-1]}[evt.action]
                dq = gymapi.Quat.from_axis_angle(gymapi.Vec3(*axis), rot_speed)
                cur_q = gymapi.Quat(*target_rot.tolist())
                new_q = dq * cur_q
                target_rot = torch.tensor([new_q.x, new_q.y, new_q.z, new_q.w], device=target_rot.device)
                goal_changed = True

            if evt.action in AXIS_MAP:
                pressed_axes.add(AXIS_MAP[evt.action])

    if goal_changed:
        gym.refresh_rigid_body_state_tensor(sim)
        cur_ee_pos = rb_states[0, ee_index_global, 0:3].clone()
        cur_ee_rot = rb_states[0, ee_index_global, 3:7].clone()
        motion_gen.start(cur_ee_pos, cur_ee_rot, target_pos.clone(), target_rot.clone())

        # 눌린 축만 FREE, 나머지 전부 LOCK
        active_axis_weights = torch.full((6,), LOCK_WEIGHT, device=dof_states.device)
        for ax in pressed_axes:
            active_axis_weights[ax] = FREE_WEIGHT

    # 이동이 끝났으면(motion_gen 비활성) 전체 축 다시 강하게 고정 → 정지 상태에서 흔들림 방지
    if not motion_gen.active:
        active_axis_weights = torch.full((6,), LOCK_WEIGHT, device=dof_states.device)

    # --- 현재 상태 읽기 ---
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    ee_pos = rb_states[0, ee_index_global, 0:3]
    ee_rot = rb_states[0, ee_index_global, 3:7]

    # --- 직선 경로 상의 "지금 순간" waypoint 가져오기 ---
    way_pos, way_quat = motion_gen.step()
    if way_pos is None:
        way_pos, way_quat = target_pos, target_rot  # 이동 중 아니면 마지막 목표 유지

    pos_err = way_pos - ee_pos

    cur_q = gymapi.Quat(ee_rot[0].item(), ee_rot[1].item(), ee_rot[2].item(), ee_rot[3].item())
    goal_q = gymapi.Quat(way_quat[0].item(), way_quat[1].item(), way_quat[2].item(), way_quat[3].item())
    err_q = goal_q * cur_q.inverse()
    if err_q.w < 0:
        err_q.x, err_q.y, err_q.z, err_q.w = -err_q.x, -err_q.y, -err_q.z, -err_q.w
    rot_err = torch.tensor([err_q.x, err_q.y, err_q.z], device=pos_err.device) * 2.0

    dpose = torch.cat([pos_err, rot_err]).unsqueeze(-1)

    j_eef = jacobian[0, ee_index_actor - 1, :, :indy7_dof_count]
    # j_eef_T = j_eef.transpose(0, 1)
    # lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
    # d_theta = j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose
    d_theta = weighted_dls(j_eef, dpose, active_axis_weights, damping)

    cur_dof = dof_states[0, :, 0]
    target_dof_pos = cur_dof + d_theta.squeeze(-1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_dof_pos))

    origin = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
    sphere_geom = gymutil.WireframeSphereGeometry(0.05*1.4, 10, 10, gymapi.Transform(p=origin), (0.0,0.0,1.0))
    gymutil.draw_lines(sphere_geom, gym, viewer, env, gymapi.Transform())
	
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)