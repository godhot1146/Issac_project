import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from indy7_controller_v1 import *
from low_amr_controller_v2 import LowAMR
from forklift_amr_controller_v1 import ForkliftAMR

# =================================================================
# 1. 시뮬레이션 및 물리 엔진 초기화
# =================================================================
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
# 3. 공장 레이아웃 스펙 (env 1개 기준, 그대로 유지)
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

room_x = 10.0         # 가로 크기 (X축)
room_y = 8.5          # 세로 크기 (Y축)
wall_height = 4.0
wall_thickness = 0.2

# =================================================================
# 4. 멀티 env 배치 설정 (세로/Y축 일렬)
# =================================================================
# NOTE: Isaac Gym의 create_env 자동 그리드(lower/upper + num_per_row)에만 의존하면
# 축 해석이 엔진 버전/설정에 따라 다르게 동작해서 env끼리 붙어버리는 문제가 있었음.
# 그래서 그리드는 "겹치지 않을 정도의 기본값"만 쓰고, 실제 간격은 아래에서
# 모든 스폰 좌표에 명시적으로 Y축 오프셋을 더해서 100% 확정적으로 제어한다.
NUM_ENVS = 3
NUM_PER_ROW = 1          # env마다 새 행(row)에 배치되도록 강제 (열 방향 오프셋 0 고정)
EXTRA_GAP_Y = 2.0        # env 사이 순수 추가 간격(m). 이 값을 키우면 간격이 더 벌어짐

# env 하나의 자연스러운 크기(원래 방 크기) 그대로 사용 -> 그리드 계산이 뒤틀리지 않게 함
env_lower = gymapi.Vec3(-room_x / 2, -room_y / 2, 0.0)
env_upper = gymapi.Vec3(room_x / 2, room_y / 2, wall_height)

# env마다 실제로 더해줄 총 Y 오프셋 = (방 한 칸 크기 + 추가 여백) * env_index
ENV_Y_STEP = room_y + EXTRA_GAP_Y

# =================================================================
# 5. 에셋 로드는 전부 "한 번만" 수행 (sim 레벨 리소스)
# =================================================================
floor_asset = gym.create_box(sim, room_x, room_y, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_y, wall_height, env_opts)
wall_y = gym.create_box(sim, room_x, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height + 0.1, env_opts)

fixed_opts = gymapi.AssetOptions()
fixed_opts.fix_base_link = True
fixed_opts.density = 100.0
wallx_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/space/wallx.urdf", fixed_opts)
wally_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/space/wally.urdf", fixed_opts)
floor_asset_v2 = gym.load_asset(sim, asset_root, "urdf/warehouse/space/floor.urdf", fixed_opts)
pillar_asset_v2 = gym.load_asset(sim, asset_root, "urdf/warehouse/space/pillar.urdf", fixed_opts)
window_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/space/window.urdf", fixed_opts)

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/move_rack/move_rack.urdf", move_rack_opts)

lock_rack_opts = gymapi.AssetOptions()
lock_rack_opts.fix_base_link = True
lock_rack_opts.density = 100.0
lock_rack_asset1 = gym.load_asset(sim, asset_root, "urdf/warehouse/rack1/rack1.urdf", move_rack_opts)
lock_rack_asset2 = gym.load_asset(sim, asset_root, "urdf/warehouse/rack2/rack2.urdf", move_rack_opts)

pallet_asset = gym.load_asset(sim, asset_root, "urdf/pallet/v2/pallet_v2.urdf", move_rack_opts)

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = True
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v2/conveyor_v2.urdf", conveyor_opts)

fixed_caser_asset = gym.load_asset(sim, asset_root, "urdf/fixed_caser/v1/fixed_caser_v1.urdf", lock_rack_opts)
taping_asset = gym.load_asset(sim, asset_root, "urdf/taping/v2/taping_v2.urdf", lock_rack_opts)

forklift_space_asset = gym.create_box(sim, 3.0, 3.0, 0.5, env_opts)

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_opts.density = 100.0
tote_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/tote/tote.urdf", tote_opts)

desk_opts = gymapi.AssetOptions()
desk_opts.fix_base_link = True
desk_opts.density = 100.0
desk_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/desk.urdf", desk_opts)

door_opts = gymapi.AssetOptions()
door_opts.fix_base_link = True
door_opts.density = 100.0
door_asset = gym.load_asset(sim, asset_root, "urdf/door/door1.urdf", door_opts)
door_asset2 = gym.load_asset(sim, asset_root, "urdf/door/door2.urdf", door_opts)

sink_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/sink.urdf", lock_rack_opts)
frame_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/frame.urdf", lock_rack_opts)
guest_desk_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/desk.urdf", lock_rack_opts)
cupboard_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/cupboard.urdf", lock_rack_opts)
chair_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/chair.urdf", lock_rack_opts)

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v2/low_amr_v2.urdf", low_opts)

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False
forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)

offset = 0.5
half_x = room_x / 2
half_y = room_y / 2
p_offset_x = half_x - 0.0
p_offset_y = half_y - 0.0

# 뷰어는 sim 레벨에서 한 번만 생성
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")


def V(x, y, z, y_off):
    """env별 Y오프셋을 더한 위치 벡터를 만드는 헬퍼."""
    return gymapi.Vec3(x, y + y_off, z)


def spawn_environment(env, env_index):
    """
    기존 1-env 스폰/배치 로직을 그대로 재현하되,
    모든 위치에 env_index * ENV_Y_STEP 만큼의 Y축 오프셋을 명시적으로 더해
    Isaac Gym 자동 그리드에 의존하지 않고 간격을 확정적으로 제어한다.
    """
    grp = env_index  # 충돌 그룹: env마다 분리해서 서로 다른 env끼리 충돌 안 하게 함
    y_off = env_index * ENV_Y_STEP

    # --- 바닥 / 외벽 / 코너 기둥 ---
    floor_h = gym.create_actor(env, floor_asset_v2, gymapi.Transform(p=V(0, 0, 0.03, y_off)), "floor", grp, 0)

    w_back = gym.create_actor(env, wallx_asset, gymapi.Transform(p=V(-half_x, 0, wall_height / 2, y_off)), "wall_back", grp, 0)
    w_front = gym.create_actor(env, window_asset, gymapi.Transform(p=V(half_x, 0, wall_height / 2, y_off)), "wall_front", grp, 0)
    w_right = gym.create_actor(env, wally_asset, gymapi.Transform(p=V(0, -half_y, wall_height / 2, y_off)), "wall_right", grp, 0)
    w_left = gym.create_actor(env, wally_asset, gymapi.Transform(p=V(0, half_y, wall_height / 2, y_off)), "wall_left", grp, 0)

    for x in [-p_offset_x, p_offset_x]:
        for y in [-p_offset_y, p_offset_y]:
            gym.create_actor(env, pillar_asset_v2, gymapi.Transform(p=V(x, y, wall_height / 2, y_off)),
                              f"corner_pillar_{env_index}_{x}_{y}", grp, 0)

    # --- 창고 구조물 ---
    gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=V(0.7, 3.4, 0.0, y_off)), f"lock_rack_asset_v1_1_{env_index}", grp, 0)
    gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=V(0.7, 2.1, 0.0, y_off)), f"lock_rack_asset_v1_2_{env_index}", grp, 0)
    gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=V(0.7, 0.8, 0.0, y_off)), f"lock_rack_asset_v1_3_{env_index}", grp, 0)

    gym.create_actor(env, lock_rack_asset2, gymapi.Transform(p=V(4.0, 3.4, 0.0, y_off)), f"lock_rack_asset_v2_1_{env_index}", grp, 0)
    gym.create_actor(env, lock_rack_asset2, gymapi.Transform(p=V(4.0, 2.1, 0.0, y_off)), f"lock_rack_asset_v2_2_{env_index}", grp, 0)
    gym.create_actor(env, lock_rack_asset2, gymapi.Transform(p=V(4.0, 0.8, 0.0, y_off)), f"lock_rack_asset_v2_3_{env_index}", grp, 0)

    gym.create_actor(env, pallet_asset, gymapi.Transform(p=V(-4.2, -3.4, 0.1, y_off)), f"pallet_asset1_{env_index}", grp, 0)
    gym.create_actor(env, pallet_asset, gymapi.Transform(p=V(-4.2, -2.1, 0.1, y_off)), f"pallet_asset2_{env_index}", grp, 0)

    gym.create_actor(env, pallet_asset, gymapi.Transform(p=V(-0.5 - offset, -2.125, 0.1, y_off)), f"pallet_asset3_{env_index}", grp, 0)

    conveyor_pos = V(2.0 - offset, -2.125, 0.05, y_off)
    conveyor_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))
    conveyor_transform = gymapi.Transform(p=conveyor_pos, r=conveyor_rot)
    gym.create_actor(env, conveyor_rack_asset, conveyor_transform, f"conveyor_asset_{env_index}", grp, 0)

    fixed_caser_pos = V(4.8 - offset, -2.125, 0.47, y_off)
    fixed_caser_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(180.0))
    fixed_caser_transform = gymapi.Transform(p=fixed_caser_pos, r=fixed_caser_rot)
    gym.create_actor(env, fixed_caser_asset, fixed_caser_transform, f"fixed_caser_asset_{env_index}", grp, 0)

    taping_pos = V(3.9 - offset, -2.125, 0.7, y_off)
    taping_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(180.0))
    taping_transform = gymapi.Transform(p=taping_pos, r=taping_rot)
    gym.create_actor(env, taping_asset, taping_transform, f"taping_asset_{env_index}", grp, 1)

    # --- 게이트/도어 ---
    gym.create_actor(env, door_asset, gymapi.Transform(p=V(-4.9 + 0.0, -0.86, 0.0, y_off)), f"door_asset1_{env_index}", grp, 0)
    gym.create_actor(env, door_asset2, gymapi.Transform(p=V(-4.9 + 0.0, 0.86, 0.0, y_off)), f"door_asset2_{env_index}", grp, 0)

    # --- 게스트 공간 ---
    r_neg90 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
    r_pos90 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))
    gym.create_actor(env, sink_asset, gymapi.Transform(p=V(-2.3, 3.7, 0.0, y_off), r=r_neg90), f"sink_asset_{env_index}", grp, 0)
    gym.create_actor(env, frame_asset, gymapi.Transform(p=V(-1.4, 3.85, 0.0, y_off), r=r_neg90), f"frame_asset_{env_index}", grp, 0)
    gym.create_actor(env, cupboard_asset, gymapi.Transform(p=V(-1.9, 4.1, 0.0, y_off), r=r_neg90), f"cupboard_asset_{env_index}", grp, 0)
    gym.create_actor(env, guest_desk_asset, gymapi.Transform(p=V(-2.3, 2.3, 0.0, y_off), r=r_neg90), f"guest_desk_asset_{env_index}", grp, 0)
    gym.create_actor(env, chair_asset, gymapi.Transform(p=V(-3.0, 2.8, 0.0, y_off), r=r_neg90), f"chair_asset1_{env_index}", grp, 0)
    gym.create_actor(env, chair_asset, gymapi.Transform(p=V(-4.0, 2.8, 0.0, y_off), r=r_neg90), f"chair_asset2_{env_index}", grp, 0)
    gym.create_actor(env, chair_asset, gymapi.Transform(p=V(-3.0, 1.8, 0.0, y_off), r=r_pos90), f"chair_asset3_{env_index}", grp, 0)
    gym.create_actor(env, chair_asset, gymapi.Transform(p=V(-4.0, 1.8, 0.0, y_off), r=r_pos90), f"chair_asset4_{env_index}", grp, 0)

    # --- 로봇 인스턴스 ---
    gym.create_actor(env, desk_asset, gymapi.Transform(p=V(-0.1, -2.6, -0.5, y_off)), f"desk_asset1_{env_index}", grp, 0)
    indy7_transform1 = gymapi.Transform(p=V(-0.1, -2.6, 0.5, y_off), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
    indy7_arm_1 = IndyArmController(
        gym, sim, env, viewer,
        asset_root=asset_root,
        urdf_path="urdf/indy_description/urdf_files/indy7_v3_vacuum.urdf",
        spawn_transform=indy7_transform1,
    )

    gym.create_actor(env, desk_asset, gymapi.Transform(p=V(4.0, -2.8, -0.7, y_off)), f"desk_asset2_{env_index}", grp, 0)
    indy7_transform2 = gymapi.Transform(p=V(4.0, -2.8, 0.2, y_off), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
    indy7_arm_2 = IndyArmController(
        gym, sim, env, viewer,
        asset_root=asset_root,
        urdf_path="urdf/indy_description/urdf_files/indy7_v3_vacuum.urdf",
        spawn_transform=indy7_transform2,
    )

    gym.create_actor(env, desk_asset, gymapi.Transform(p=V(2.35, -1.4, -0.5, y_off)), f"desk_asset3_{env_index}", grp, 0)
    indy7_transform3 = gymapi.Transform(p=V(2.35, -1.4, 0.5, y_off), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
    indy7_arm_3 = IndyArmController(
        gym, sim, env, viewer,
        asset_root=asset_root,
        urdf_path="urdf/indy_description/urdf_files/indy7_v3_vacuum.urdf",
        spawn_transform=indy7_transform3,
    )

    low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=V(2.35, 0.0, 0.3, y_off)), f"low_asset_{env_index}", grp, 1)
    low_amr = LowAMR(gym, sim, env, low_handle)

    forklift_handle = gym.create_actor(env, forklift_asset, gymapi.Transform(p=V(-3.2, -2.125, 0.3, y_off)), f"forklift_asset_{env_index}", grp, 1)
    forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

    return {
        "y_off": y_off,
        "indy7_arm_1": indy7_arm_1,
        "indy7_arm_2": indy7_arm_2,
        "indy7_arm_3": indy7_arm_3,
        "low_amr": low_amr,
        "forklift_amr": forklift_amr,
    }


# =================================================================
# 6. env 3개 생성 (Y축 명시적 오프셋으로 세로 일렬 배치) + 동일 스폰 로직 반복
# =================================================================
envs = []
env_controllers = []  # env_index -> dict(controllers)

for i in range(NUM_ENVS):
    env = gym.create_env(sim, env_lower, env_upper, NUM_PER_ROW)
    envs.append(env)
    ctrls = spawn_environment(env, i)
    env_controllers.append(ctrls)

# env가 Y축으로 ENV_Y_STEP 간격씩 세로로 늘어서 있으므로,
# 전체 행렬 중앙(가운데 env의 y_off)을 바라보되 X축 뒤쪽 멀리 + 높은 곳에서 조망
center_y = env_controllers[NUM_ENVS // 2]["y_off"]
gym.viewer_camera_look_at(
    viewer, envs[0],
    gymapi.Vec3(-2.56, 11.21, 15.09),
    gymapi.Vec3(-1.16, 10.41, 12.56),
)

# =================================================================
# 7. 텐서 준비 (모든 env 생성 후 1회)
# =================================================================
gym.prepare_sim(sim)

for ctrls in env_controllers:
    ctrls["indy7_arm_1"].setup_tensors()
    ctrls["indy7_arm_2"].setup_tensors()
    ctrls["indy7_arm_3"].setup_tensors()

# waypoint_list / forklift_waypoint_list의 좌표는 각 env 내부 컨트롤러가
# 자기 env의 로컬 좌표계(자기 actor 기준 상대/역기구학 등)로 해석하므로
# env마다 동일한 리스트를 그대로 재사용한다.
waypoint_list = [
    (2.35, 2.1, '+x', 'BACKWARD'),
    (4.0, 2.1, '+x', 'FORWARD'),
    (2.35, 2.1, '+y', 'BACKWARD'),
    (2.35, -0.3, '+y', 'BACKWARD')
]

forklift_waypoint_list = [
    (-3.0, -2.125, '+x', 'FORWARD'),
    (-2.0, -2.125, '+x', 'FORWARD'),
    (-2.5, -2.125, '-y', 'BACKWARD'),
    (-2.6027, -0.086084, '-x', 'BACKWARD')
]

for ctrls in env_controllers:
    ctrls["low_amr"].set_state(0)
    ctrls["forklift_amr"].set_state(0)

# ===============================
##### simulation part #####
# ===============================
frame_count = 0

while not gym.query_viewer_has_closed(viewer):
    for event in gym.query_viewer_action_events(viewer):
        if event.value > 0:
            if event.action == "start_process":
                cam_matrix = gym.get_viewer_camera_transform(viewer, envs[0])
                cam_pos = cam_matrix.p
                r = cam_matrix.r
                qx, qy, qz, qw = r.x, r.y, r.z, r.w
                forward_x = 2.0 * (qx * qz + qw * qy)
                forward_y = 2.0 * (qy * qz - qw * qx)
                forward_z = 1.0 - 2.0 * (qx * qx + qy * qy)
                target_dist = 3.0
                target_x = cam_pos.x + forward_x * target_dist
                target_y = cam_pos.y + forward_y * target_dist
                target_z = cam_pos.z + forward_z * target_dist
                print(f"\ngym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3({cam_pos.x:.2f}, {cam_pos.y:.2f}, {cam_pos.z:.2f}), gymapi.Vec3({target_x:.2f}, {target_y:.2f}, {target_z:.2f}))")

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # 3개 env 전부에 대해 동일한 FSM 로직 반복 실행
    for ctrls in env_controllers:
        ctrls["low_amr"].lift_and_locate_fsm_v3(
            frame_count=frame_count,
            waypoint_list=waypoint_list
        )
        ctrls["forklift_amr"].lift_and_locate_fsm(
            frame_count=frame_count,
            waypoint_list=forklift_waypoint_list,
            lift_target_pos=0.1,
            lift_threshold=0.015
        )

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)