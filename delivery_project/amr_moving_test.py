import os
import csv
import time
import numpy as np
from isaacgym import gymapi, gymutil
from low_amr_controller_v2 import LowAMR
from forklift_amr_controller_v1 import ForkliftAMR

# =================================================================
# 🎯 로깅 활성화 여부 (맨 위 플래그) — False로 두면 CSV 파일 생성/기록을 완전히 건너뜁니다.
# =================================================================
ENABLE_LOGGING = False

# 🎯 접촉력(Net Contact Force) 로깅을 위한 텐서 API
#    torch가 없거나 acquire에 실패해도 나머지 로깅은 그대로 동작하도록 방어적으로 처리합니다.
try:
    import torch
    from isaacgym import gymtorch
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[경고] torch/gymtorch 로드 실패 ➡️ 접촉력 로깅은 비활성화됩니다. ({e})")
    TORCH_AVAILABLE = False

# =================================================================
# 1. 시뮬레이션 및 물리 엔진 초기화
# =================================================================
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Dual-AMR Teleop DEBUG v2 (접촉력/토크/가속도 전방위 로깅)")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12
sim_params.physx.num_velocity_iterations = 8

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

gym.set_light_parameters(sim, 0, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1.0, 0.0, -1.0))

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

# =================================================================
# 2. 공장 레이아웃 (외벽/기둥)
# =================================================================
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/isaac_assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

room_size = 20.0
wall_height = 4.0
wall_thickness = 0.2

floor_asset = gym.create_box(sim, room_size, room_size, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_size, wall_height, env_opts)
wall_y = gym.create_box(sim, room_size, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height, env_opts)

env = gym.create_env(sim, gymapi.Vec3(-room_size / 2, -room_size / 2, 0), gymapi.Vec3(room_size / 2, room_size / 2, room_size), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

half_r = room_size / 2
w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_r, 0, wall_height / 2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_r, 0, wall_height / 2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_r, wall_height / 2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_r, wall_height / 2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
    gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

p_offset = half_r - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)
for x in [-p_offset, p_offset]:
    for y in [-p_offset, p_offset]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height / 2)), f"corner_pillar_{x}_{y}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

pillar_side_positions = np.arange(-2.5, 3.5, 2.5)
for y_pos in pillar_side_positions:
    for x_pos in [-p_offset, p_offset]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height / 2)), f"side_pillar_{x_pos}_{y_pos}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

##### ==================== #####
# --- 로봇 1: Low-AMR 스폰 ---
##### ==================== #####

low_move_rack_opts = gymapi.AssetOptions()
low_move_rack_opts.fix_base_link = False
low_move_rack_opts.density = 100.0
low_move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/rack2/rack2.urdf", low_move_rack_opts)
low_move_rack_handle = gym.create_actor(env, low_move_rack_asset, gymapi.Transform(p=gymapi.Vec3(-3.0, -2.0, 0.05)), "low_move_rack_asset", -1, 0)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_opts.replace_cylinder_with_capsule = True
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v2/low_amr_v2.urdf", low_opts)
low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(-3.0, 0.0, 1.0)), "low_asset", -1, 1)

low_amr = LowAMR(gym, sim, env, low_handle)

low_amr.set_lift_rotation(0.0)
low_amr.start_lift_yaw_lock(target_local_angle=0.0)

##### ==================== #####
# --- 로봇 2: Forklift-AMR 스폰 ---
##### ==================== #####

pallet_opts = gymapi.AssetOptions()
pallet_opts.fix_base_link = False
pallet_opts.density = 100.0
pallet_asset = gym.load_asset(sim, asset_root, "urdf/pallet/v1/pallet_test.urdf", pallet_opts)
pallet_handle = gym.create_actor(env, pallet_asset, gymapi.Transform(p=gymapi.Vec3(6.0, 0.0, 0.3)), "pallet_handle", -1, 0)

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False
forklift_opts.replace_cylinder_with_capsule = True
forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)
forklift_handle = gym.create_actor(env, forklift_asset, gymapi.Transform(p=gymapi.Vec3(3.0, 0.0, 0.3)), "forklift_asset", -1, 1)

forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

# =========================================================================
# 3. 🎯 접촉력 텐서 준비 (반드시 모든 액터 생성 후, prepare_sim 이후에 acquire)
# =========================================================================
gym.prepare_sim(sim)

CONTACT_LOGGING_AVAILABLE = False
contact_force_np = None
_net_contact_force_tensor = None

if ENABLE_LOGGING and TORCH_AVAILABLE:
    try:
        _net_contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
        CONTACT_LOGGING_AVAILABLE = True
        print("[진단] 접촉력(Net Contact Force) 텐서 획득 성공 ➡️ 접촉력 로깅 활성화")
    except Exception as e:
        print(f"[경고] 접촉력 텐서 획득 실패 ➡️ 접촉력 로깅 비활성화 ({e})")
        CONTACT_LOGGING_AVAILABLE = False


def refresh_contact_forces():
    """매 프레임 1회만 호출해서 접촉력 텐서를 갱신하고 numpy 배열로 캐싱합니다."""
    global contact_force_np
    if not CONTACT_LOGGING_AVAILABLE:
        return
    gym.refresh_net_contact_force_tensor(sim)
    t = gymtorch.wrap_tensor(_net_contact_force_tensor)
    contact_force_np = t.cpu().numpy()  # shape: (전체 rigid body 수, 3)


def body_contact_force_mag(body_idx):
    """DOMAIN_SIM 기준 rigid body 인덱스로 접촉력 크기(N)를 반환합니다."""
    if not CONTACT_LOGGING_AVAILABLE or body_idx == -1 or contact_force_np is None:
        return 0.0
    f = contact_force_np[body_idx]
    return float(np.linalg.norm(f))


def find_body_indices(env, handle, name_list):
    """이름 리스트에 대응하는 DOMAIN_SIM rigid body 인덱스 리스트를 반환합니다."""
    idxs = []
    for name in name_list:
        idx = gym.find_actor_rigid_body_index(env, handle, name, gymapi.DOMAIN_SIM)
        if idx != -1:
            idxs.append(idx)
    return idxs


# 로봇별 주요 링크의 DOMAIN_SIM 인덱스 캐싱
low_left_wheel_body = gym.find_actor_rigid_body_index(env, low_handle, "left_wheel_link", gymapi.DOMAIN_SIM)
low_right_wheel_body = gym.find_actor_rigid_body_index(env, low_handle, "right_wheel_link", gymapi.DOMAIN_SIM)
low_caster_bodies = find_body_indices(env, low_handle, low_amr.CASTER_LINK_NAMES)
low_lift_bodies = find_body_indices(env, low_handle, low_amr.LIFT_LINK_NAMES)

fork_left_wheel_body = gym.find_actor_rigid_body_index(env, forklift_handle, "left_wheel_link", gymapi.DOMAIN_SIM)
fork_right_wheel_body = gym.find_actor_rigid_body_index(env, forklift_handle, "right_wheel_link", gymapi.DOMAIN_SIM)
fork_caster_bodies = find_body_indices(env, forklift_handle, forklift_amr.CASTER_LINK_NAMES)
fork_lift_bodies = find_body_indices(env, forklift_handle, forklift_amr.LIFT_LINK_NAMES)

# 🎯 실제 모터 토크 로깅용 effort 한계값 캐싱 (포화 여부 판단용)
low_dof_props = gym.get_actor_dof_properties(env, low_handle)
low_effort_limits = np.array(low_dof_props['effort'], dtype=np.float32)
fork_dof_props = gym.get_actor_dof_properties(env, forklift_handle)
fork_effort_limits = np.array(fork_dof_props['effort'], dtype=np.float32)

# =========================================================================
# 4. 텔레옵 제어 상태 변수
# =========================================================================

SEL_LOW = "LOW"
SEL_FORKLIFT = "FORKLIFT"
selected_robot = SEL_LOW

speed_cfg = {
    SEL_LOW: {"lin": 0.5, "ang": 1.0, "lift": 0.05},
    SEL_FORKLIFT: {"lin": 0.5, "ang": 1.0, "lift": 0.05},
}
SPEED_STEP_RATIO = 1.1
LIFT_SPEED_STEP_RATIO = 1.2

MIN_LIN_SPEED, MAX_LIN_SPEED = 0.05, 2.0
MIN_ANG_SPEED, MAX_ANG_SPEED = 0.1, 4.0
MIN_LIFT_SPEED, MAX_LIFT_SPEED = 0.01, 0.3

lift_target = {SEL_LOW: 0.0, SEL_FORKLIFT: 0.0}
LIFT_RANGE = {
    SEL_LOW: (low_amr.LIFT_MIN_HEIGHT, low_amr.LIFT_MAX_HEIGHT),
    SEL_FORKLIFT: (forklift_amr.LIFT_MIN_HEIGHT, forklift_amr.LIFT_MAX_HEIGHT),
}

DT = 1.0 / 60.0

held = {
    "UP": False, "DOWN": False, "LEFT": False, "RIGHT": False,
    "LIFT_UP": False, "LIFT_DOWN": False,
}

# =========================================================================
# 5. 진단 로거 설정 (ENABLE_LOGGING이 False면 파일 생성/기록을 건너뜁니다)
# =========================================================================
LOG_EVERY_N_FRAMES = 1
YAW_SPIKE_WARN_DEG = 3.0
WHEEL_VEL_ERR_WARN = 3.0
LIFT_ROTATE_ERR_WARN_DEG = 5.0
CONTACT_FORCE_WARN_N = 50.0     # 리프트/캐스터 접촉력이 이 값(N) 이상이면 "바닥에 끌림" 의심 경고
FORCE_SATURATION_WARN_RATIO = 0.95  # 모터 토크가 effort 한계의 95% 이상이면 포화 경고

log_fields = [
    "frame", "sim_time", "robot",
    "cmd_lin", "cmd_ang",
    "ramped_v_lin", "ramped_v_ang",
    "wheel_L_target_rad_s", "wheel_R_target_rad_s",
    "wheel_L_actual_rad_s", "wheel_R_actual_rad_s",
    "wheel_L_err", "wheel_R_err",
    "wheel_L_force_Nm", "wheel_R_force_Nm",
    "wheel_L_force_ratio", "wheel_R_force_ratio",
    "wheel_L_contact_N", "wheel_R_contact_N",
    "caster_contact_sum_N", "lift_contact_sum_N",
    "base_x", "base_y", "base_yaw_deg",
    "d_yaw_deg",
    "actual_vx", "actual_vy", "actual_wz_deg_s",
    "accel_x", "accel_y", "ang_accel_deg_s2",
    "lift_target_m", "lift_actual_m",
    "lift_force_N", "lift_force_ratio",
    "lift_rotate_target_deg", "lift_rotate_actual_deg", "lift_rotate_err_deg",
    "lift_rotate_force_Nm", "lift_rotate_force_ratio",
    "caster_L_angle_deg", "caster_R_angle_deg",
    "caster_L_force_Nm", "caster_R_force_Nm",
    "frame_wall_dt_ms",
]

log_file = None
log_writer = None
log_path = None

if ENABLE_LOGGING:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"teleop_debug_v2_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    print(f"[진단 로그] 기록 시작 ➡️ {log_path}")
else:
    print("[진단 로그] ENABLE_LOGGING=False ➡️ 로그 파일을 생성하지 않습니다.")

prev_yaw = {SEL_LOW: None, SEL_FORKLIFT: None}
prev_vel = {SEL_LOW: (0.0, 0.0, 0.0), SEL_FORKLIFT: (0.0, 0.0, 0.0)}  # (vx, vy, wz_rad)
prev_wall_time = time.perf_counter()


def get_yaw(quat):
    siny_cosp = 2.0 * (quat['w'] * quat['z'] + quat['x'] * quat['y'])
    cosy_cosp = 1.0 - 2.0 * (quat['y'] ** 2 + quat['z'] ** 2)
    return np.arctan2(siny_cosp, cosy_cosp)


def force_ratio(force_val, idx, effort_limits):
    if idx == -1 or effort_limits[idx] <= 0:
        return 0.0
    return abs(float(force_val)) / float(effort_limits[idx])


def collect_robot_diag(name, controller, cmd_lin_val, cmd_ang_val,
                        left_wheel_body, right_wheel_body, caster_bodies, lift_bodies,
                        effort_limits, is_forklift=False):
    body_states = gym.get_actor_rigid_body_states(controller.env, controller.handle, gymapi.STATE_ALL)
    pos = body_states['pose']['p'][0]
    rot = body_states['pose']['r'][0]
    lin_vel = body_states['vel']['linear'][0]
    ang_vel = body_states['vel']['angular'][0]
    yaw = get_yaw(rot)

    dof_states = gym.get_actor_dof_states(controller.env, controller.handle, gymapi.STATE_ALL)
    dof_forces = gym.get_actor_dof_forces(controller.env, controller.handle)  # 실제 적용된 힘/토크

    wheel_L_target = float(controller.dof_velocity_targets[controller.left_wheel_idx]) if controller.left_wheel_idx != -1 else 0.0
    wheel_R_target = float(controller.dof_velocity_targets[controller.right_wheel_idx]) if controller.right_wheel_idx != -1 else 0.0
    wheel_L_actual = float(dof_states['vel'][controller.left_wheel_idx]) if controller.left_wheel_idx != -1 else 0.0
    wheel_R_actual = float(dof_states['vel'][controller.right_wheel_idx]) if controller.right_wheel_idx != -1 else 0.0

    wheel_L_force = float(dof_forces[controller.left_wheel_idx]) if controller.left_wheel_idx != -1 else 0.0
    wheel_R_force = float(dof_forces[controller.right_wheel_idx]) if controller.right_wheel_idx != -1 else 0.0
    wheel_L_force_ratio = force_ratio(wheel_L_force, controller.left_wheel_idx, effort_limits)
    wheel_R_force_ratio = force_ratio(wheel_R_force, controller.right_wheel_idx, effort_limits)

    wheel_L_contact = body_contact_force_mag(left_wheel_body)
    wheel_R_contact = body_contact_force_mag(right_wheel_body)
    caster_contact_sum = sum(body_contact_force_mag(b) for b in caster_bodies)
    lift_contact_sum = sum(body_contact_force_mag(b) for b in lift_bodies)

    lift_actual = controller.get_lift_height()
    lift_target_val = lift_target[name]
    lift_force = float(dof_forces[controller.lift_joint_idx]) if controller.lift_joint_idx != -1 else 0.0
    lift_force_ratio = force_ratio(lift_force, controller.lift_joint_idx, effort_limits)

    lift_rot_target_deg = 0.0
    lift_rot_actual_deg = 0.0
    lift_rot_err_deg = 0.0
    lift_rot_force = 0.0
    lift_rot_force_ratio = 0.0
    caster_L_angle_deg = 0.0
    caster_R_angle_deg = 0.0
    caster_L_force = 0.0
    caster_R_force = 0.0

    if not is_forklift:
        if controller.lift_rotate_joint_idx != -1:
            lift_rot_target_deg = float(np.degrees(controller.dof_position_targets[controller.lift_rotate_joint_idx]))
            lift_rot_actual_deg = float(np.degrees(controller.get_lift_rotation()))
            lift_rot_err_deg = lift_rot_target_deg - lift_rot_actual_deg
            lift_rot_force = float(dof_forces[controller.lift_rotate_joint_idx])
            lift_rot_force_ratio = force_ratio(lift_rot_force, controller.lift_rotate_joint_idx, effort_limits)
    else:
        if controller.left_caster_idx != -1:
            caster_L_angle_deg = float(np.degrees(dof_states['pos'][controller.left_caster_idx]))
            caster_L_force = float(dof_forces[controller.left_caster_idx])
        if controller.right_caster_idx != -1:
            caster_R_angle_deg = float(np.degrees(dof_states['pos'][controller.right_caster_idx]))
            caster_R_force = float(dof_forces[controller.right_caster_idx])

    d_yaw_deg = 0.0
    if prev_yaw[name] is not None:
        d_yaw = np.arctan2(np.sin(yaw - prev_yaw[name]), np.cos(yaw - prev_yaw[name]))
        d_yaw_deg = float(np.degrees(d_yaw))
    prev_yaw[name] = yaw

    vx, vy, wz = float(lin_vel['x']), float(lin_vel['y']), float(ang_vel['z'])
    prev_vx, prev_vy, prev_wz = prev_vel[name]
    accel_x = (vx - prev_vx) / DT
    accel_y = (vy - prev_vy) / DT
    ang_accel_deg_s2 = np.degrees(wz - prev_wz) / DT
    prev_vel[name] = (vx, vy, wz)

    diag = {
        "robot": name,
        "cmd_lin": cmd_lin_val, "cmd_ang": cmd_ang_val,
        "ramped_v_lin": controller.current_v_lin, "ramped_v_ang": controller.current_v_ang,
        "wheel_L_target_rad_s": wheel_L_target, "wheel_R_target_rad_s": wheel_R_target,
        "wheel_L_actual_rad_s": wheel_L_actual, "wheel_R_actual_rad_s": wheel_R_actual,
        "wheel_L_err": wheel_L_target - wheel_L_actual, "wheel_R_err": wheel_R_target - wheel_R_actual,
        "wheel_L_force_Nm": wheel_L_force, "wheel_R_force_Nm": wheel_R_force,
        "wheel_L_force_ratio": wheel_L_force_ratio, "wheel_R_force_ratio": wheel_R_force_ratio,
        "wheel_L_contact_N": wheel_L_contact, "wheel_R_contact_N": wheel_R_contact,
        "caster_contact_sum_N": caster_contact_sum, "lift_contact_sum_N": lift_contact_sum,
        "base_x": float(pos['x']), "base_y": float(pos['y']), "base_yaw_deg": float(np.degrees(yaw)),
        "d_yaw_deg": d_yaw_deg,
        "actual_vx": vx, "actual_vy": vy, "actual_wz_deg_s": float(np.degrees(wz)),
        "accel_x": accel_x, "accel_y": accel_y, "ang_accel_deg_s2": ang_accel_deg_s2,
        "lift_target_m": lift_target_val, "lift_actual_m": lift_actual,
        "lift_force_N": lift_force, "lift_force_ratio": lift_force_ratio,
        "lift_rotate_target_deg": lift_rot_target_deg, "lift_rotate_actual_deg": lift_rot_actual_deg,
        "lift_rotate_err_deg": lift_rot_err_deg,
        "lift_rotate_force_Nm": lift_rot_force, "lift_rotate_force_ratio": lift_rot_force_ratio,
        "caster_L_angle_deg": caster_L_angle_deg, "caster_R_angle_deg": caster_R_angle_deg,
        "caster_L_force_Nm": caster_L_force, "caster_R_force_Nm": caster_R_force,
    }
    return diag


# =========================================================================
# 6. 뷰어 생성 및 키보드 이벤트 구독
# =========================================================================

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 6, 8), gymapi.Vec3(0, 0, 0))

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "SELECT_LOW")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_9, "SELECT_FORKLIFT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "MOVE_FORWARD")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "MOVE_BACKWARD")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "TURN_LEFT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "TURN_RIGHT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Z, "LIFT_DOWN")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "LIFT_UP")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT_BRACKET, "SPEED_DOWN")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT_BRACKET, "SPEED_UP")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_COMMA, "LIFT_SPEED_DOWN")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_PERIOD, "LIFT_SPEED_UP")

frame_count = 0

print("=" * 70)
print(" [Dual-AMR Teleop DEBUG v2] 접촉력/토크/가속도 전방위 로깅")
print("  0 / 9 : 선택 | ↑↓ 전후진 | ←→ 회전 | Z/X 리프트 | [ ] 이동속도 | , . 리프트속도")
if ENABLE_LOGGING:
    print(f"  로그 파일 : {log_path}")
else:
    print("  로그 : 비활성화 (ENABLE_LOGGING=False)")
print("=" * 70)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


while not gym.query_viewer_has_closed(viewer):
    now_wall = time.perf_counter()
    frame_wall_dt_ms = (now_wall - prev_wall_time) * 1000.0
    prev_wall_time = now_wall

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    for evt in gym.query_viewer_action_events(viewer):
        pressed = evt.value > 0

        if evt.action == "SELECT_LOW" and pressed:
            selected_robot = SEL_LOW
            print(f"\n[선택] ➡️ Low-AMR 로 전환")
        elif evt.action == "SELECT_FORKLIFT" and pressed:
            selected_robot = SEL_FORKLIFT
            print(f"\n[선택] ➡️ Forklift-AMR 로 전환")
        elif evt.action == "MOVE_FORWARD":
            held["UP"] = pressed
        elif evt.action == "MOVE_BACKWARD":
            held["DOWN"] = pressed
        elif evt.action == "TURN_LEFT":
            held["LEFT"] = pressed
        elif evt.action == "TURN_RIGHT":
            held["RIGHT"] = pressed
        elif evt.action == "LIFT_UP":
            held["LIFT_UP"] = pressed
        elif evt.action == "LIFT_DOWN":
            held["LIFT_DOWN"] = pressed
        elif evt.action == "SPEED_UP" and pressed:
            cfg = speed_cfg[selected_robot]
            cfg["lin"] = clip(cfg["lin"] * SPEED_STEP_RATIO, MIN_LIN_SPEED, MAX_LIN_SPEED)
            cfg["ang"] = clip(cfg["ang"] * SPEED_STEP_RATIO, MIN_ANG_SPEED, MAX_ANG_SPEED)
            print(f"[{selected_robot}] 이동 속도 증가 ➡️ lin={cfg['lin']:.3f} ang={cfg['ang']:.3f}")
        elif evt.action == "SPEED_DOWN" and pressed:
            cfg = speed_cfg[selected_robot]
            cfg["lin"] = clip(cfg["lin"] / SPEED_STEP_RATIO, MIN_LIN_SPEED, MAX_LIN_SPEED)
            cfg["ang"] = clip(cfg["ang"] / SPEED_STEP_RATIO, MIN_ANG_SPEED, MAX_ANG_SPEED)
            print(f"[{selected_robot}] 이동 속도 감소 ➡️ lin={cfg['lin']:.3f} ang={cfg['ang']:.3f}")
        elif evt.action == "LIFT_SPEED_UP" and pressed:
            cfg = speed_cfg[selected_robot]
            cfg["lift"] = clip(cfg["lift"] * LIFT_SPEED_STEP_RATIO, MIN_LIFT_SPEED, MAX_LIFT_SPEED)
            print(f"[{selected_robot}] 리프트 속도 증가 ➡️ {cfg['lift']:.3f}")
        elif evt.action == "LIFT_SPEED_DOWN" and pressed:
            cfg = speed_cfg[selected_robot]
            cfg["lift"] = clip(cfg["lift"] / LIFT_SPEED_STEP_RATIO, MIN_LIFT_SPEED, MAX_LIFT_SPEED)
            print(f"[{selected_robot}] 리프트 속도 감소 ➡️ {cfg['lift']:.3f}")

    cur_cfg = speed_cfg[selected_robot]
    cmd_lin = 0.0
    cmd_ang = 0.0
    if held["UP"]:
        cmd_lin += cur_cfg["lin"]
    if held["DOWN"]:
        cmd_lin -= cur_cfg["lin"]
    if held["LEFT"]:
        cmd_ang += cur_cfg["ang"]
    if held["RIGHT"]:
        cmd_ang -= cur_cfg["ang"]

    low_cmd_lin = cmd_lin if selected_robot == SEL_LOW else 0.0
    low_cmd_ang = cmd_ang if selected_robot == SEL_LOW else 0.0
    fork_cmd_lin = cmd_lin if selected_robot == SEL_FORKLIFT else 0.0
    fork_cmd_ang = cmd_ang if selected_robot == SEL_FORKLIFT else 0.0

    low_amr.set_twist_velocity(low_cmd_lin, low_cmd_ang)
    forklift_amr.set_twist_velocity(fork_cmd_lin, fork_cmd_ang)

    lo_h, hi_h = LIFT_RANGE[selected_robot]
    if held["LIFT_UP"]:
        lift_target[selected_robot] = clip(lift_target[selected_robot] + cur_cfg["lift"] * DT, lo_h, hi_h)
    if held["LIFT_DOWN"]:
        lift_target[selected_robot] = clip(lift_target[selected_robot] - cur_cfg["lift"] * DT, lo_h, hi_h)

    low_amr.set_lift_height(lift_target[SEL_LOW])
    forklift_amr.set_lift_height(lift_target[SEL_FORKLIFT])

    low_amr._update_lift_yaw_lock()

    low_amr.apply_actuator_commands()
    forklift_amr.apply_actuator_commands()

    # -----------------------------------------------------------------
    # 🎯 접촉력 텐서는 프레임당 1회만 갱신 (두 로봇이 공유) — 로깅이 켜져 있을 때만 의미가 있음
    # -----------------------------------------------------------------
    if ENABLE_LOGGING:
        refresh_contact_forces()

    # -----------------------------------------------------------------
    # 진단 로그 기록 (ENABLE_LOGGING=False면 전체 스킵)
    # -----------------------------------------------------------------
    if ENABLE_LOGGING and frame_count % LOG_EVERY_N_FRAMES == 0:
        sim_time = frame_count * DT

        low_diag = collect_robot_diag(
            SEL_LOW, low_amr, low_cmd_lin, low_cmd_ang,
            low_left_wheel_body, low_right_wheel_body, low_caster_bodies, low_lift_bodies,
            low_effort_limits, is_forklift=False,
        )
        fork_diag = collect_robot_diag(
            SEL_FORKLIFT, forklift_amr, fork_cmd_lin, fork_cmd_ang,
            fork_left_wheel_body, fork_right_wheel_body, fork_caster_bodies, fork_lift_bodies,
            fork_effort_limits, is_forklift=True,
        )

        for diag in (low_diag, fork_diag):
            row = {"frame": frame_count, "sim_time": round(sim_time, 4), "frame_wall_dt_ms": round(frame_wall_dt_ms, 3)}
            row.update(diag)
            log_writer.writerow(row)

            if abs(diag["d_yaw_deg"]) > YAW_SPIKE_WARN_DEG:
                print(f"⚠️ [{diag['robot']}] Yaw 스파이크! Δ{diag['d_yaw_deg']:.2f}° (frame={frame_count})")
            if abs(diag["wheel_L_err"]) > WHEEL_VEL_ERR_WARN or abs(diag["wheel_R_err"]) > WHEEL_VEL_ERR_WARN:
                print(f"⚠️ [{diag['robot']}] 바퀴 속도 오차 큼: L={diag['wheel_L_err']:.2f} R={diag['wheel_R_err']:.2f} (frame={frame_count})")
            if diag["robot"] == SEL_LOW and abs(diag["lift_rotate_err_deg"]) > LIFT_ROTATE_ERR_WARN_DEG:
                print(f"⚠️ [LOW] 상판 카운터회전 오차 큼: {diag['lift_rotate_err_deg']:.2f}° (frame={frame_count})")
            # if diag["lift_contact_sum_N"] > CONTACT_FORCE_WARN_N:
            #     print(f"⚠️ [{diag['robot']}] 리프트/화물판 접촉력 큼 ➡️ 바닥에 끌리고 있을 가능성! "
            #           f"{diag['lift_contact_sum_N']:.1f}N (frame={frame_count}, lift_actual={diag['lift_actual_m']:.4f}m)")
            if diag["wheel_L_force_ratio"] > FORCE_SATURATION_WARN_RATIO or diag["wheel_R_force_ratio"] > FORCE_SATURATION_WARN_RATIO:
                print(f"⚠️ [{diag['robot']}] 구동륜 토크 포화 임박! L_ratio={diag['wheel_L_force_ratio']:.2f} "
                      f"R_ratio={diag['wheel_R_force_ratio']:.2f} (frame={frame_count})")

    if ENABLE_LOGGING and frame_count % 300 == 0:
        log_file.flush()

    if frame_count % 120 == 0:
        sel_cfg = speed_cfg[selected_robot]
        cur_height = (low_amr.get_lift_height() if selected_robot == SEL_LOW else forklift_amr.get_lift_height())
        print(f"[Status] 선택: {selected_robot} | lin={sel_cfg['lin']:.2f} ang={sel_cfg['ang']:.2f} "
              f"liftSpd={sel_cfg['lift']:.3f} | LiftTarget={lift_target[selected_robot]:.3f}m "
              f"CurrentHeight={cur_height:.4f}m | frame_dt={frame_wall_dt_ms:.2f}ms | "
              f"contact_logging={'ON' if CONTACT_LOGGING_AVAILABLE else 'OFF'}")

    gym.sync_frame_time(sim)
    frame_count += 1

if ENABLE_LOGGING:
    log_file.close()
    print(f"[진단 로그] 저장 완료 ➡️ {log_path}")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)