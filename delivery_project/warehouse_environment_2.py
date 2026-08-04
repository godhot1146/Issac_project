import os
import glob
import numpy as np
from isaacgym import gymapi, gymutil
from indy7_controller_v1 import *
from low_amr_controller_v2 import LowAMR, LowAmrMotionStep, LowAmrStepRunner
from forklift_amr_controller_v1 import ForkliftAMR, ForkliftMotionStep, ForkliftStepRunner
from cardboard_box_manager import CardboardBoxManager, BoxStageRegistry
from conveyor_belt import ConveyorBelt, ConveyorMotionStep, ConveyorStepRunner
from scipy.spatial.transform import Rotation as R

# ============================================================================
# [SECTION 1] 시뮬레이션 / 물리 엔진 초기화
# ============================================================================
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

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)


# ============================================================================
# [SECTION 2] 공장 레이아웃 (바닥/벽/기둥) 에셋 정의 및 환경 생성
# ============================================================================
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/isaac_assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

room_x = 10.0
room_y = 8.5
wall_height = 4.0
wall_thickness = 0.2

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

env = gym.create_env(sim, gymapi.Vec3(-room_x/2, -room_y/2, 0), gymapi.Vec3(room_x/2, room_y/2, wall_height), 1)

floor_h = gym.create_actor(env, floor_asset_v2, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)

half_x = room_x / 2
half_y = room_y / 2

w_back  = gym.create_actor(env, wallx_asset, gymapi.Transform(p=gymapi.Vec3(-half_x, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, window_asset, gymapi.Transform(p=gymapi.Vec3(half_x, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wally_asset, gymapi.Transform(p=gymapi.Vec3(0, -half_y, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wally_asset, gymapi.Transform(p=gymapi.Vec3(0, half_y, wall_height/2)), "wall_left", 0, 0)

p_offset_x = half_x - 0.0
p_offset_y = half_y - 0.0
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset_x, p_offset_x]:
    for y in [-p_offset_y, p_offset_y]:
        p_handle = gym.create_actor(env, pillar_asset_v2, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)), f"corner_pillar_{x}_{y}", 0, 0)


# ============================================================================
# [SECTION 3] 창고 구조물 스폰 (랙, 팔레트, 컨베이어, 공정 설비, 사무 공간 소품)
# ============================================================================
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)
dummy_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/dummy.urdf", box_opts)
package_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/package/package.urdf", box_opts)

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/move_rack/move_rack.urdf", move_rack_opts)

lock_rack_opts = gymapi.AssetOptions()
lock_rack_opts.fix_base_link = True
lock_rack_opts.density = 100.0
lock_rack_asset1 = gym.load_asset(sim, asset_root, "urdf/warehouse/rack1/rack1.urdf", move_rack_opts)
lock_rack_asset2 = gym.load_asset(sim, asset_root, "urdf/warehouse/rack2/rack2.urdf", move_rack_opts)
lock_rack_asset2_test = gym.load_asset(sim, asset_root, "urdf/warehouse/rack2/rack2_test.urdf", move_rack_opts)

rack_v1_1 = gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=gymapi.Vec3(0.7, 3.4, 0.0)), "lock_rack_asset_v1_1", -1, 0)
rack_v1_1 = gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=gymapi.Vec3(0.7, 2.1, 0.0)), "lock_rack_asset_v1_1", -1, 0)
rack_v1_1 = gym.create_actor(env, lock_rack_asset1, gymapi.Transform(p=gymapi.Vec3(0.7, 0.8, 0.0)), "lock_rack_asset_v1_1", -1, 0)

rack_v2_1 = gym.create_actor(env, lock_rack_asset2, gymapi.Transform(p=gymapi.Vec3(4.0, 3.4, 0.0)), "lock_rack_asset_v2_1", -1, 0)
rack_v2_2 = gym.create_actor(env, lock_rack_asset2_test, gymapi.Transform(p=gymapi.Vec3(4.0, 2.1, 0.0)), "lock_rack_asset_v2_2", -1, 0)

package_handler1 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0, 2.1+0.4, 1.07)), "package_asset", -1, 0)
package_handler2 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0-0.35, 2.1+0.4, 1.07)), "package_asset", -1, 0)
package_handler3 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0+0.35, 2.1+0.4, 1.07)), "package_asset", -1, 0)
package_handler4 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0, 2.1+0.0, 1.07)), "package_asset", -1, 0)
package_handler5 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0-0.35, 2.1+0.0, 1.07)), "package_asset", -1, 0)
package_handler6 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0+0.35, 2.1+0.0, 1.07)), "package_asset", -1, 0)
package_handler7 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0, 2.1-0.4, 1.07)), "package_asset", -1, 0)
package_handler8 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0-0.35, 2.1-0.4, 1.07)), "package_asset", -1, 0)
package_handler9 = gym.create_actor(env, package_asset, gymapi.Transform(p=gymapi.Vec3(4.0+0.35, 2.1-0.4, 1.07)), "package_asset", -1, 0)

package_handlers = [
    package_handler1, package_handler2, package_handler3,
    package_handler4, package_handler5, package_handler6,
    package_handler7, package_handler8, package_handler9,
]

rack_v2_3 = gym.create_actor(env, lock_rack_asset2, gymapi.Transform(p=gymapi.Vec3(4.0, 0.8, 0.0)), "lock_rack_asset_v2_3", -1, 0)

pallet_asset = gym.load_asset(sim, asset_root, "urdf/pallet/v2/pallet_v2.urdf", move_rack_opts)
pallet_handle1 = gym.create_actor(env, pallet_asset, gymapi.Transform(p=gymapi.Vec3(-4.2, -3.4, 0.1)), "pallet_asset1", -1, 0)
pallet_handle2 = gym.create_actor(env, pallet_asset, gymapi.Transform(p=gymapi.Vec3(-4.2, -2.1, 0.1)), "pallet_asset2", -1, 0)

conveyor_opts = gymapi.AssetOptions()
conveyor_opts.fix_base_link = True
conveyor_opts.density = 100.0
conveyor_rack_asset = gym.load_asset(sim, asset_root, "urdf/conveyor/v2/conveyor_v2.urdf", conveyor_opts)

offset = 0.5
pallet_handle4 = gym.create_actor(env, pallet_asset, gymapi.Transform(p=gymapi.Vec3(-0.5 - offset, -2.125, 0.1)), "pallet_asset3", -1, 0)

conveyor_pos = gymapi.Vec3(2.0-offset, -2.125, 0.05)
conveyor_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))
conveyor_transform = gymapi.Transform(p=conveyor_pos, r=conveyor_rot)
conveyor_handle = gym.create_actor(env, conveyor_rack_asset, conveyor_transform, "conveyor_asset", -1, 0)

fixed_caser_asset = gym.load_asset(sim, asset_root, "urdf/fixed_caser/v1/fixed_caser_v1.urdf", lock_rack_opts)
fixed_caser_pos = gymapi.Vec3(4.8-offset, -2.125, 0.47)
fixed_caser_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(180.0))
fixed_caser_transform = gymapi.Transform(p=fixed_caser_pos, r=fixed_caser_rot)
fixed_caser_handle = gym.create_actor(env, fixed_caser_asset, fixed_caser_transform, "fixed_caser_asset", -1, 2)

taping_asset = gym.load_asset(sim, asset_root, "urdf/taping/v2/taping_v2.urdf", lock_rack_opts)
taping_pos = gymapi.Vec3(3.9-offset, -2.125, 0.7)
taping_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(180.0))
taping_transform = gymapi.Transform(p=taping_pos, r=taping_rot)
taping_handle = gym.create_actor(env, taping_asset, taping_transform, "taping_asset", -1, 2)

fold_up_asset = gym.load_asset(sim, asset_root, "urdf/fold_up/fold_up.urdf", lock_rack_opts)
fold_up_pos = gymapi.Vec3(0.7, -2.125, 1.1)
fold_up_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(180.0))
fold_up_transform = gymapi.Transform(p=fold_up_pos, r=fold_up_rot)
fold_up_handle = gym.create_actor(env, fold_up_asset, fold_up_transform, "fold_upasset", -1, 1)

box_station_opts = gymapi.AssetOptions()
box_station_opts.fix_base_link = True
box_station_opts.density = 100.0
box_station_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box_station/box_station.urdf", box_station_opts)

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_opts.density = 100.0
tote_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/tote/tote.urdf", tote_opts)

desk_opts = gymapi.AssetOptions()
desk_opts.fix_base_link = True
desk_opts.density = 100.0
desk_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/desk.urdf", desk_opts)
desk_narrow_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/desk_narrow.urdf", desk_opts)

door_opts = gymapi.AssetOptions()
door_opts.fix_base_link = True
door_opts.density = 100.0
door_asset = gym.load_asset(sim, asset_root, "urdf/door/door1.urdf", door_opts)
door_asset2 = gym.load_asset(sim, asset_root, "urdf/door/door2.urdf", door_opts)

door_handle1 = gym.create_actor(env, door_asset, gymapi.Transform(p=gymapi.Vec3(-4.9 + 0.0, -0.86, 0.0)), "door_asset", -1, 0)
door_handle2 = gym.create_actor(env, door_asset2, gymapi.Transform(p=gymapi.Vec3(-4.9 + 0.0, 0.86, 0.0)), "door_asset", -1, 0)

sink_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/sink.urdf", lock_rack_opts)
frame_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/frame.urdf", lock_rack_opts)
guest_desk_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/desk.urdf", lock_rack_opts)
cupboard_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/cupboard.urdf", lock_rack_opts)
chair_asset = gym.load_asset(sim, asset_root, "urdf/guest_space/v1/chair.urdf", lock_rack_opts)

sink_handle = gym.create_actor(env, sink_asset, gymapi.Transform(p=gymapi.Vec3(-2.3, 3.7, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "sink_asset", -1, 0)
frame_handle = gym.create_actor(env, frame_asset, gymapi.Transform(p=gymapi.Vec3(-1.4, 3.85, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "frame_asset", -1, 0)
cupboard_handle = gym.create_actor(env, cupboard_asset, gymapi.Transform(p=gymapi.Vec3(-1.9, 4.1, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "cupboard_asset", -1, 0)
guest_desk_handle = gym.create_actor(env, guest_desk_asset, gymapi.Transform(p=gymapi.Vec3(-2.3, 2.3, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "guest_desk_asset", -1, 0)
chair_handle1 = gym.create_actor(env, chair_asset, gymapi.Transform(p=gymapi.Vec3(-3.0, 2.8, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "chair_asset1", -1, 0)
chair_handle2 = gym.create_actor(env, chair_asset, gymapi.Transform(p=gymapi.Vec3(-4.0, 2.8, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "chair_asset2", -1, 0)
chair_handle3 = gym.create_actor(env, chair_asset, gymapi.Transform(p=gymapi.Vec3(-3.0, 1.8, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))), "chair_asset3", -1, 0)
chair_handle4 = gym.create_actor(env, chair_asset, gymapi.Transform(p=gymapi.Vec3(-4.0, 1.8, 0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))), "chair_asset4", -1, 0)


# ============================================================================
# [SECTION 4] 로봇 인스턴스 스폰 
# ============================================================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.37, -0.77, 3.90), gymapi.Vec3(2.08, 1.08, 1.65))

arm_opts = gymapi.AssetOptions()
arm_opts.fix_base_link = True
arm_opts.density = 100.0
arm_asset = gym.load_asset(sim, asset_root, "urdf/indy_description/urdf_files/indy7_v3_vacuum.urdf", arm_opts)

desk_handle1 = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(-0.1, -2.6, -0.5)), "desk_asset1", -1, 0)
indy7_transform1 = gymapi.Transform(p=gymapi.Vec3(-0.1, -2.6, 0.5), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
indy7_arm_1 = IndyArmController(gym, sim, env, viewer, arm_asset, spawn_transform=indy7_transform1, actor_name="indy7_arm_1")

desk_handle2 = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(4.0, -2.8, -0.6)), "desk_asset2", -1, 0)
indy7_transform2 = gymapi.Transform(p=gymapi.Vec3(4.0, -2.8, 0.4), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
indy7_arm_2 = IndyArmController(gym, sim, env, viewer, arm_asset, spawn_transform=indy7_transform2, actor_name="indy7_arm_2")

desk_handle3 = gym.create_actor(env, desk_narrow_asset, gymapi.Transform(p=gymapi.Vec3(2.35, -1.4 -0.35, 0.01)), "desk_asset3", -1, 0)
indy7_transform3 = gymapi.Transform(p=gymapi.Vec3(2.35, -1.4 -0.35, 1.01), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
indy7_arm_3 = IndyArmController(gym, sim, env, viewer, arm_asset, spawn_transform=indy7_transform3, actor_name="indy7_arm_3")

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_opts.replace_cylinder_with_capsule = True
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v2/low_amr_v2.urdf", low_opts)
low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(2.35, 0.0, 0.3)), "low_asset", -1, 1)
low_amr = LowAMR(gym, sim, env, low_handle)

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False
forklift_opts.replace_cylinder_with_capsule = True
forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)
forklift_handle = gym.create_actor(env, forklift_asset, gymapi.Transform(p=gymapi.Vec3(-3.2, -2.125, 0.3)), "forklift_asset", -1, 1)
forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

frame_paths = sorted(glob.glob(os.path.join(asset_root, "urdf/conveyor/v2/anim/*.png")))
conveyor = ConveyorBelt(
    gym, sim, env, conveyor_handle,
    belt_link_name="belt_link",
    texture_frame_paths=frame_paths,
    speed=0.5,
)

# ============================================================================
# [SECTION 5] 박스 상태(Stage) 레지스트리 등록
# ============================================================================
registry = BoxStageRegistry()

registry.register("unfolded_flat", {
    "joint_right_to_front": 10.0,
    "joint_right_to_back":  10.0,
    "joint_front_to_left": -10.0,
})

registry.register("fold_sides", {
    "joint_right_to_front": 90.0,
    "joint_right_to_back":  90.0,
    "joint_front_to_left": -90.0,
})

registry.register(
    "close_fb_bottom",
    joint_targets_deg={"joint_front_to_down": -90.0, "joint_back_to_down": 90.0},
    base="fold_sides",
)

registry.register(
    "close_rl_bottom",
    joint_targets_deg={"joint_right_to_down": -90.0, "joint_left_to_down": 90.0},
    base="close_fb_bottom",
    lock_joints=["joint_right_to_down", "joint_left_to_down",
                 "joint_front_to_down", "joint_back_to_down"],
)

registry.register(
    "close_top",
    joint_targets_deg={
        "joint_right_to_up": 90.0, "joint_left_to_up": -90.0,
        "joint_front_to_up": 90.0, "joint_back_to_up": -90.0,
    },
    base="close_rl_bottom",
    lock_joints=["joint_right_to_up", "joint_left_to_up",
                 "joint_front_to_up", "joint_back_to_up"],
)

registry.register("close_side_bottom_half", {
    "joint_right_to_down": -50.0,
    "joint_left_to_down":   50.0,
})


# ============================================================================
# [SECTION 6] 카드보드 박스 스폰
# ============================================================================
cardboard_opts = gymapi.AssetOptions()
cardboard_opts.fix_base_link = False
cardboard_opts.density = 100.0
cardboard_asset = gym.load_asset(sim, asset_root, "urdf/cardboard_box/v2/box_v2.urdf", cardboard_opts)

box_positions = [
    gymapi.Vec3(4.15, -3.47, 1.28), gymapi.Vec3(4.15, -3.53, 1.28),
    gymapi.Vec3(4.15, -3.59, 1.28), gymapi.Vec3(4.15, -3.65, 1.28),
    gymapi.Vec3(4.15, -3.71, 1.28), gymapi.Vec3(4.15, -3.77, 1.28),
    gymapi.Vec3(4.15, -3.83, 1.28), gymapi.Vec3(4.15, -3.89, 1.28),
]

box_managers = []
for i, pos in enumerate(box_positions, start=1):
    box_pose = gymapi.Transform(p=pos, r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0)))
    box_handle = gym.create_actor(env, cardboard_asset, box_pose, f"cordboard_box_asset{i}", -1, 2)
    box_manager = CardboardBoxManager(gym, sim, env, box_handle, stage_registry=registry, joint_speed=1.2)
    box_managers.append(box_manager)

box_station_handle2 = gym.create_actor(env, box_station_asset, gymapi.Transform(p=gymapi.Vec3(4.0, -3.7, 0.72), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))), "box_station_asset", -1, 0)


# ============================================================================
# [SECTION 7] 텐서 준비 (gym.prepare_sim 이후 1회)
# ============================================================================
gym.prepare_sim(sim)
indy7_arm_1.setup_tensors()
indy7_arm_2.setup_tensors()
indy7_arm_3.setup_tensors()
conveyor.setup_tensors(belt_length=3.5)
box_managers[0].setup_tensors()

conveyor.register_item(box_managers[0].handle)


# ============================================================================
# [SECTION 8] 로봇 프리셋 등록 (흡착 대상, 관절 포즈 이름 매핑)
# ============================================================================
indy7_arm_1.register_attachable(box_managers[0].handle)
indy7_arm_1.register_joint_pose("init_pose", [-2.0255, 0.2495, 0.9428, -0.0014, 1.9548, 0.4432])

indy7_arm_2.register_attachable(box_managers[0].handle)
indy7_arm_2.register_joint_pose("init_pose", [-1.2403, 0.3089, -1.7372, -1.0972, -0.3799, -0.5353])
indy7_arm_2.register_joint_pose("move_box3", [1.7970, 0.3498, -1.7833, -1.1369, -0.3526, -0.4891])
indy7_arm_2.register_joint_pose("fold_box1", [1.4619, -0.1843, -0.7517, 0.1897, -0.6595, -1.7569])
indy7_arm_2.register_joint_pose("fold_box4", [0.8073, -0.8635, 0.3811, 0.8210, -1.2486, -2.2129])
indy7_arm_2.register_joint_pose("fold_box7", [0.7465, -1.2928, 0.9982, 0.8441, -1.3903, -1.9818])
indy7_arm_2.register_joint_pose("fold_box9", [0.7458, -1.4048, 1.1457, 0.8384, -1.4135, -1.7979])

for h in package_handlers:
    indy7_arm_3.register_attachable(h)
indy7_arm_3.register_joint_pose("init_pose", [-1.571, 0.0, -1.571, 0.0, -1.571, 0.0])   ## -x 방향에 적재할 경우
indy7_arm_3.register_joint_pose("init_pose2", [1.571, 0.0, 1.571, 0.0, 1.571, 0.0])     ## +x 방향에 적재할 경우
indy7_arm_3.register_joint_pose("pick_pose2", [-1.571, 0.0, 1.571, 0.0, 1.571, 0.0])
indy7_arm_3.register_joint_pose("pick_pose", [1.571, 0.0, -1.571, 0.0, -1.571, 0.0])
indy7_arm_3.register_joint_pose("place_pose1", [-0.8767, 0.2263, -1.7686, 0.0012, -1.6044, 0.6765])
indy7_arm_3.register_joint_pose("place_pose2", [1.5700, -0.4539, 1.8892, -0.0012, 1.7009, -0.0003])

arms = [indy7_arm_1, indy7_arm_2, indy7_arm_3]
selected_arm_idx = 0


# ============================================================================
# [SECTION 9] 스텝 콜백 함수 정의 (on_enter / on_complete / wait_for에서 사용)
# ============================================================================
def _detach(ctx):
    ctx.arm.detach()

def _attach(ctx):
    ctx.arm.attach_nearest(distance_threshold=0.3)

def _box_side_fold(ctx):
    ctx.box_managers.apply_stage("fold_sides")

def _box_bottom_fold1(ctx):
    ctx.box_managers.apply_stage("close_fb_bottom")

def _box_bottom_fold2(ctx):
    ctx.box_managers.apply_stage("close_fb_bottom")

def _box_bottom_fold3(ctx):
    ctx.box_managers.apply_stage("close_fb_bottom")

def _box_side_bottom_fold1(ctx):
    ctx.box_managers.apply_stage("close_rl_bottom")

def _box_fix1(ctx):
    ctx.box_managers.lock_joint("joint_back_to_down")
    ctx.box_managers.lock_joint("joint_front_to_down")

def _release_from_conveyor(ctx):
    """상자를 컨베이어 관리에서 완전히 제외 -> 중력/마찰 등 순수 물리로 복귀."""
    ctx.conveyor.unregister_item(ctx.box_managers.handle)
    print("[Conveyor] box_manager1 컨베이어 등록 해제 -> 자연 물리 복귀")

def _attach_to_pallet(ctx):
    ctx.box_managers.attach_to_platform(ctx.pallet_handle)

def _mark_forklift_ready(ctx):
    ctx.forklift_start["start"] = True

def _mark_low_amr_done(ctx):
    ctx.low_amr_state["done"] = True

def _finalize_box_before_second_move(ctx):
    ctx.box_managers.scan_and_capture(package_handlers, lambda h: get_actor_world_pos(gym, env, h))
    ctx.box_managers.apply_stage("close_top")
    ctx.box_managers.lock_children_offsets()

def _mark_second_move_done(ctx):
    ctx.conveyor_state["second_move_done"] = True

def get_actor_world_pos(gym, env, handle):
    body_states = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
    pos = body_states['pose']['p'][0]  # 첫 번째(=루트) 링크 기준
    return pos['x'], pos['y'], pos['z']


# ============================================================================
# [SECTION 10] 자율 시퀀스 러너 등록
#   - 이 구역은 "무엇을 언제 할지"만 선언한다. 실제 동작은 메인 루프에서
#     각 러너의 update(frame_count)가 호출될 때 발생한다.
# ============================================================================

# ---- 10-1. LowAMR 시퀀스 ---------------------------------------------------
low_amr_steps = [
    LowAmrMotionStep(
        "MOVE_TO_CARGO",
        target_x=4.0, target_y=2.1,
        axis_order='y',
        initial_facing='+y', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x',
        lift_height=0.0,
        reset_lift_plate=True,
        lift_yaw_lock=False,
        hold_seconds=1.0,
    ),
    LowAmrMotionStep(
        "LIFT_UP",
        target_x=4.0, target_y=2.1,
        axis_order='x',
        initial_facing='+x', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x',
        lift_height=0.16,
        lift_yaw_lock=True,
        hold_seconds=1.0,
    ),
    LowAmrMotionStep(
        "RETREAT_WITH_CARGO",
        target_x=2.45, target_y=2.1,
        axis_order='x',
        initial_facing='+x', direction1='BACKWARD',
        mid_facing='+x', direction2='BACKWARD',
        final_facing='+x',
        lift_height=0.16,
        lift_yaw_lock=True,
        hold_seconds=1.0,
    ),
    LowAmrMotionStep(
        "MOVE_TO_DEST",
        target_x=2.45, target_y=-0.9,
        axis_order='x',
        initial_facing='+x', direction1='BACKWARD',
        mid_facing='-y', direction2='FORWARD',
        final_facing='-y',
        lift_height=0.16,
        lift_yaw_lock=True,
        hold_seconds=0.1,
        on_complete=[_mark_low_amr_done],
    ),
]
low_amr_task_state = {"done": False}
low_amr_ctx = StepContext(low_amr_state=low_amr_task_state)
low_amr_runner = LowAmrStepRunner(low_amr, low_amr_steps, context=low_amr_ctx)

# ---- 10-2. Forklift 시퀀스 --------------------------------------------------
forklift_start_state = {"start": False}
forklift_steps = [
    ForkliftMotionStep(
        "INITIAL",
        target_x=-3.2, target_y=-2.125,
        axis_order='x',
        initial_facing='+x', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x',
        lift_height=-0.03,
        hold_seconds=0.1,
        wait_for=lambda ctx: ctx.forklift_start["start"],
    ),
    ForkliftMotionStep(
        "MOVE_TO_SHELF_UNDER_AND_LIFT",
        target_x=-2.05, target_y=-2.125,
        axis_order='x',
        initial_facing='+x', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x',
        lift_height=0.16,
        hold_seconds=0.1,
    ),
    ForkliftMotionStep(
        "MOVE_SHELF",
        target_x=-2.5, target_y=-0.5,
        axis_order='x',
        initial_facing='+x', direction1='BACKWARD',
        mid_facing='+y', direction2='FORWARD',
        final_facing='-x',
        lift_height=-0.03,
        hold_seconds=0.1,
    ),
    ForkliftMotionStep(
        "MOVE_TO_SHELF_UNDER",
        target_x=-1.3, target_y=-1.0,
        axis_order='x',
        initial_facing='-x', direction1='BACKWARD',
        mid_facing='-y', direction2='FORWARD',
        final_facing='-y',
        lift_height=-0.03,
        hold_seconds=0.1,
    ),
]
forklift_ctx = StepContext(forklift_start=forklift_start_state)
forklift_runner = ForkliftStepRunner(forklift_amr, forklift_steps, context=forklift_ctx)

# ---- 10-3. Indy7 팔 시퀀스 (컨텍스트 정의) ----------------------------------
conveyor_task_state = {"second_move_done": False}

ctx1 = StepContext(
    arm=indy7_arm_1,
    box_managers=box_managers[0],
    conveyor_state=conveyor_task_state,
    conveyor=conveyor,
    pallet_handle=pallet_handle4,
    forklift_start=forklift_start_state,
)
ctx2 = StepContext(
    arm=indy7_arm_2,
    box_managers=box_managers[0],
)
ctx3 = StepContext(
    arm=indy7_arm_3,
    low_amr_state=low_amr_task_state,
)

linear_speed_value = 0.5

# ---- 10-3-1. runner1: 컨베이어 픽업 → 팔레트 이적 ---------------------------
steps1 = [
    ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
    ArmMotionStep(
        "MOVE_TO_PICK", "cartesian_delta", delta_axis=0, delta_value=0.24,
        linear_speed=linear_speed_value, on_complete=[],
        wait_for=lambda ctx: ctx.conveyor_state["second_move_done"]
    ),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=1, delta_value=-0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=2, delta_value=-0.12,
                  linear_speed=linear_speed_value, on_complete=[_attach]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=2, delta_value=0.06,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=0, delta_value=-0.70,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=2, delta_value=-0.33,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=0, delta_value=-0.22,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=2, delta_value=-0.31,
                  linear_speed=linear_speed_value, on_complete=[_detach]),
    ArmMotionStep("MOVE_TO_PICK", "cartesian_delta", delta_axis=2, delta_value=0.30,
                  linear_speed=linear_speed_value,
                  on_complete=[_detach, _release_from_conveyor, _attach_to_pallet, _mark_forklift_ready]),
]

# ---- 10-3-2. runner2: 박스 접기 ---------------------------------------------
steps2 = [
    ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
    ArmMotionStep("MOVE_TO_BOX1", "cartesian_delta", delta_axis=1, delta_value=-0.32,
                  linear_speed=linear_speed_value, on_complete=[_attach]),
    ArmMotionStep("MOVE_TO_BOX2", "cartesian_delta", delta_axis=1, delta_value=0.32,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_BOX3", "joint", target="move_box3"),
    ArmMotionStep("FOLD_BOX1", "joint", target="fold_box1", on_complete=[_box_side_fold]),
    ArmMotionStep("FOLD_BOX2", "cartesian_delta", delta_axis=2, delta_value=-0.08,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("FOLD_BOX3", "cartesian_delta", delta_axis=0, delta_value=0.3,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("FOLD_BOX4", "joint", target="fold_box4"),
    ArmMotionStep("FOLD_BOX5", "cartesian_delta", delta_axis=2, delta_value=-0.12,
                  linear_speed=linear_speed_value, on_start=[_box_bottom_fold1], on_complete=[]),
    ArmMotionStep("FOLD_BOX6", "cartesian_delta", delta_axis=0, delta_value=0.05,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("FOLD_BOX7", "joint", target="fold_box7"),
    ArmMotionStep("FOLD_BOX8", "cartesian_delta", delta_axis=2, delta_value=-0.04,
                  linear_speed=linear_speed_value, on_start=[_box_bottom_fold2], on_complete=[]),
    ArmMotionStep("FOLD_BOX9", "joint", target="fold_box9"),
    ArmMotionStep("FOLD_BOX10", "cartesian_delta", delta_axis=0, delta_value=-0.10,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("FOLD_BOX11", "cartesian_delta", delta_axis=2, delta_value=-0.065,
                  linear_speed=linear_speed_value, on_start=[_box_bottom_fold3], on_complete=[]),
    ArmMotionStep("FOLD_BOX12", "cartesian_delta", delta_axis=1, delta_value=0.01,
                  linear_speed=linear_speed_value, on_start=[_box_bottom_fold3], on_complete=[]),
    ArmMotionStep("FOLD_BOX13", "cartesian_delta", delta_axis=0, delta_value=-1.15,
                  linear_speed=linear_speed_value, on_start=[_box_side_bottom_fold1],
                  on_complete=[_box_fix1, _detach]),
    ArmMotionStep("FOLD_BOX14", "cartesian_delta", delta_axis=1, delta_value=-0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
]

# ---- 10-3-3. runner3: 패키지 적재 -------------------------------------------
steps3 = [
    ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
    ArmMotionStep("MOVE_TO_PICK1", "joint", target="pick_pose",
                  wait_for=lambda ctx: ctx.low_amr_state["done"]),
    ArmMotionStep("MOVE_TO_PICK2", "cartesian_delta", delta_axis=0, delta_value=-0.125,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK3", "cartesian_delta", delta_axis=1, delta_value=0.1,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PICK4", "cartesian_delta", delta_axis=2, delta_value=-0.241,
                  linear_speed=linear_speed_value, on_complete=[_attach]),
    ArmMotionStep("MOVE_TO_READY1", "cartesian_delta", delta_axis=2, delta_value=0.241,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_READY2", "joint", target="pick_pose"),
    ArmMotionStep("MOVE_TO_READY3", "cartesian_delta", delta_axis=0, delta_value=-0.2,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_READY4", "cartesian_delta", delta_axis=1, delta_value=-0.03,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_READY5", "joint", target="place_pose1"),
    ArmMotionStep("MOVE_TO_READY6", "joint", target="init_pose"),
    ArmMotionStep("MOVE_TO_PLACE1", "cartesian_delta", delta_axis=0, delta_value=-0.13,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE2", "cartesian_delta", delta_axis=1, delta_value=-0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE3", "cartesian_delta", delta_axis=2, delta_value=-0.6,
                  linear_speed=linear_speed_value, on_complete=[_detach]),
    ArmMotionStep("MOVE_TO_PLACE4", "cartesian_delta", delta_axis=2, delta_value=0.6,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE5", "cartesian_delta", delta_axis=1, delta_value=0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE6", "cartesian_delta", delta_axis=0, delta_value=0.13,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_OTHER_PICK1", "joint", target="init_pose2"),
    ArmMotionStep("MOVE_TO_OTHER_PICK2", "joint", target="pick_pose2"),
    ArmMotionStep("MOVE_TO_OTHER_PICK2", "cartesian_delta", delta_axis=0, delta_value=-0.041,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_OTHER_PICK3", "cartesian_delta", delta_axis=1, delta_value=0.1,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_OTHER_PICK4", "cartesian_delta", delta_axis=2, delta_value=-0.241,
                  linear_speed=linear_speed_value, on_complete=[_attach]),
    ArmMotionStep("MOVE_TO_READY1", "cartesian_delta", delta_axis=2, delta_value=0.241,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_READY2", "joint", target="pick_pose2"),
    ArmMotionStep("MOVE_TO_READY3", "cartesian_delta", delta_axis=1, delta_value=-0.2,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_READY4", "joint", target="place_pose2"),
    ArmMotionStep("MOVE_TO_READY5", "joint", target="init_pose2"),
    ArmMotionStep("MOVE_TO_PLACE1", "cartesian_delta", delta_axis=0, delta_value=-0.31,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE2", "cartesian_delta", delta_axis=1, delta_value=-0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE3", "cartesian_delta", delta_axis=2, delta_value=-0.45,
                  linear_speed=linear_speed_value, on_complete=[_detach]),
    ArmMotionStep("MOVE_TO_PLACE4", "cartesian_delta", delta_axis=2, delta_value=0.45,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE5", "cartesian_delta", delta_axis=1, delta_value=0.02,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_PLACE6", "cartesian_delta", delta_axis=0, delta_value=0.31,
                  linear_speed=linear_speed_value, on_complete=[]),
    ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
]

runner1 = ArmStepRunner(indy7_arm_1, steps1, context=ctx1)
runner2 = ArmStepRunner(indy7_arm_2, steps2, context=ctx2)
runner3 = ArmStepRunner(indy7_arm_3, steps3, context=ctx3)

# ---- 10-4. Conveyor 시퀀스 ---------------------------------------------------
conveyor_ctx = StepContext(
    conveyor=conveyor,
    runner2=runner2,
    runner3=runner3,
    box_managers=box_managers[0],
    conveyor_state=conveyor_task_state,
)

conveyor_steps = [
    ConveyorMotionStep(
        "FIRST_MOVE",
        distance=1.0,
        direction='FORWARD',
        wait_for=lambda ctx: ctx.runner2.done,
    ),
    ConveyorMotionStep(
        "SECOND_MOVE",
        distance=2.0,
        direction='FORWARD',
        wait_for=lambda ctx: ctx.runner3.done,
        on_enter=[_finalize_box_before_second_move],
        on_complete=[_mark_second_move_done],
    ),
]
conveyor_runner = ConveyorStepRunner(conveyor, conveyor_steps, context=conveyor_ctx)


# ============================================================================
# [SECTION 11] 키 입력 등록
# ============================================================================
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "start_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_process")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "select_prev_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "select_next_joint")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "decrease_joint_angle")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "increase_joint_angle")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "box_stage_0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "box_stage_1")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_7, "box_stage_2")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_8, "box_stage_3")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "toggle_auto_rotation")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_manual_mode")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "toggle_box_manual_mode")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "box_move_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "box_move_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "box_move_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "box_move_right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "box_move_up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "box_move_down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_9, "select_prev_arm")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "select_next_arm")


# ============================================================================
# [SECTION 12] 메인 시뮬레이션 루프
# ============================================================================
frame_count = 0
while not gym.query_viewer_has_closed(viewer):

    # ---- 12-1. 입력 이벤트 처리 --------------------------------------------
    events = gym.query_viewer_action_events(viewer)
    for event in events:
        if event.value > 0:
            if event.action == "start_process":
                cam_matrix = gym.get_viewer_camera_transform(viewer, env)
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
                print(f"\ngym.viewer_camera_look_at(viewer, env, gymapi.Vec3({cam_pos.x:.2f}, {cam_pos.y:.2f}, {cam_pos.z:.2f}), gymapi.Vec3({target_x:.2f}, {target_y:.2f}, {target_z:.2f}))")

                for i, handle in enumerate(package_handlers, start=1):
                    x, y, z = get_actor_world_pos(gym, env, handle)
                    print(f"package_handler{i} 위치: ({x:.4f}, {y:.4f}, {z:.4f})")

            elif event.action == "select_prev_arm":
                selected_arm_idx = (selected_arm_idx - 1) % len(arms)
                print(f"[선택된 팔] indy7_arm_{selected_arm_idx + 1}")

            elif event.action == "select_next_arm":
                selected_arm_idx = (selected_arm_idx + 1) % len(arms)
                print(f"[선택된 팔] indy7_arm_{selected_arm_idx + 1}")

    # ---- 12-2. 물리 스텝 ----------------------------------------------------
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)

    # ---- 12-3. 수동 조작시 팔 3대 처리 (키 입력 / IK·JOINT 계산 / 흡착 추종 / 마커) ----
    for i, arm in enumerate(arms):
        if i == selected_arm_idx:
            arm.process_keyboard_input(events)
        arm.step()
        arm.update_attachment()
        arm.draw_target_marker()

        if i == selected_arm_idx:
            arm.draw_ee_marker(color=(0.0, 1.0, 0.0))
        else:
            arm.draw_ee_marker()

    # ---- 12-4. 자율 시퀀스 러너 업데이트 (SECTION 10에서 등록한 러너들) --------
    if not low_amr_runner.done:
        low_amr_runner.update(frame_count)

    if not forklift_runner.done:
        forklift_runner.update(frame_count)

    if not runner1.done:
        runner1.update(frame_count)

    if not runner2.done:
        runner2.update(frame_count)

    if not runner3.done:
        runner3.update(frame_count)

    if not conveyor_runner.done:
        conveyor_runner.update(frame_count)

    # ---- 12-5. DOF 명령 일괄 반영 -------------------------------------------
    flush_dof_targets(gym, sim, arms)

    # ---- 12-6. 박스 매니저 후처리 (전체 박스, 매 프레임) ------------------------
    for box_manager in box_managers:
        box_manager.update_joints()
        box_manager.update_platform_lock()
        box_manager.update_children()

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)