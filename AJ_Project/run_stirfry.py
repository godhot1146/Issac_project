"""
run_stirfry.py — 두산 A0509 볶음 공정 씬 + 키보드 다중모드 제어

씬: 볶음 도면을 기준으로 Bonitkit + V2 조리/준비 테이블 + 그릇 11개를
    A0509 작업 반경 안에 배치한다. A0509는 전용 stand 상단 z=0.81에 고정 장착한다.
제어: 실행 중 키로 모드를 바꿔가며 직접 조작.

  ┌─────────────── 키 맵 ───────────────┐
  │ [모드]  1: JSC(관절)  2: TSC(좌표+IK)  3: OSC(좌표+동역학)
  │ [좌표이동 · TSC/OSC]  W/S: X±   A/D: Y±   Q/E: Z±
  │ [관절이동 · JSC]      J/L: 관절 선택   U/O: 선택관절 각도±
  │ [공통]  R: 홈 자세    (뷰어 창 닫기: 종료)
  └──────────────────────────────────────┘

제어 로직은 controllers/doosan_controller.py (JSC/TSC/OSC 통합).
좌표 목표는 '로봇 베이스 기준'. OSC는 월드 텐서를 쓰므로 장착높이(+0.81) 보정해 넘긴다.

실행:  conda activate issac_env  &&  python run_stirfry.py
       (OSC용 gymtorch→ninja 필요)
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil   # torch보다 먼저

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers"))
from doosan_controller import DoosanController

from asset_config import get_asset_root
asset_root = get_asset_root()   # 컴퓨터마다 에셋 위치 자동 탐색/저장 (asset_config.py 참고)

BASE_Z     = 0.81     # A0509_Stand.step의 장착 상판 높이
CART_STEP  = 0.01     # 좌표 목표 이동 스텝(m)
JOINT_STEP = 0.05     # 관절 이동 스텝(rad)
HOME_Q     = np.array([0.0, 0.0, 1.2, 0.0, 1.0, 0.0], dtype=np.float32)

# 볶음 도면 배치. 새 stand(0.6 x 0.9 m)와 각 설비 사이에 최소
# 0.15 m의 여유를 두면서, 모든 그릇 중심을 베이스에서 0.83 m 안에 둔다.
BONITKIT_POS       = (0.0, 1.07, 0.0)
COMPLETE_TABLE_POS = (-0.625, 0.10, 0.0)
PREPARE_TABLE_POS  = (0.10, -0.25, 0.0)
TABLE_YAW_DEG      = 90.0
TABLE_TOP_Z        = 0.85
# STEP/explicit collision 기준 첫 무간섭 높이보다 약 1 mm 높게 시작한다.
# 이전 5~8.5 mm 낙하 간격을 줄여 초기 접촉 충격은 낮추되, 겹친 채 생성하지 않는다.
COOK_BOWL_Z        = 0.8225
INGREDIENT_BOWL_Z  = 0.8255
TABLE_ASSET_VERSION = "v2"
A0509_STAND_URDF    = "urdf/a0509_stand/a0509_stand.urdf"
COMPLETE_TABLE_URDF = "urdf/complete_table/complete_table.urdf"
PREPARE_TABLE_URDF  = "urdf/prepare_table/prepare_table.urdf"
BOWL_URDF            = "urdf/stirfry_bowl/stirfry_bowl.urdf"
A0509_URDF           = "urdf/doosan_a0509/a0509.urdf"
A0509_GRIPPER_URDF   = "urdf/a0509_stirfry_gripper/a0509_stirfry_gripper.urdf"
GRIPPER_BODY_NAME    = "stirfry_gripper_link"

# 실측값이 없으므로 dry metal contact의 보수적인 시작값이다. grasp 실험 전에
# 재질/표면 상태에 맞춰 조정할 수 있는 tunable simulation parameter로 취급한다.
BOWL_FRICTION       = 0.50
GRIPPER_FRICTION    = 0.50
TABLE_FRICTION      = 0.50
CONTACT_RESTITUTION = 0.0
CONTACT_OFFSET      = 0.001  # 1 mm: 0.5~1.0 mm 설계 clearance보다 과도하지 않게 설정
REST_OFFSET         = 0.0

BOWL_COLLISION_SHAPES = 129
GRIPPER_COLLISION_SHAPES = 156
COMPLETE_TABLE_COLLISION_SHAPES = 77  # 72 top prisms + 5 lower-frame boxes
PREPARE_TABLE_COLLISION_SHAPES = 405  # 397 top prisms + 8 lower-frame boxes

# V2 STEP 원점 기준 실제 홀 중심. V2에서도 중심은 기존과 동일하다.
# 조리 그릇은 Ø250 mm, 재료 그릇은 도면의 Ø200 mm 제한보다 작은
# Ø187.5 mm(0.75배)로 사용해 200 mm 간격의 이웃 그릇과 겹치지 않게 한다.
COMPLETE_BOWL_LOCAL_XY = (0.25, 0.0)
PREPARE_BOWL_LOCAL_XY = (
    *((-0.475, y) for y in (-0.30, -0.10, 0.10, 0.30, 0.50)),
    *((x, -0.475) for x in (-0.30, -0.10, 0.10, 0.30, 0.50)),
)
INGREDIENT_BOWL_SCALE = 0.75


def pose(x, y, z=0.0, yaw_deg=0.0):
    """Z축 yaw를 포함한 actor pose를 만든다."""
    rotation = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0, 0, 1), np.radians(yaw_deg)
    )
    return gymapi.Transform(p=gymapi.Vec3(x, y, z), r=rotation)


def local_xy_to_world(local_xy, origin_xy, yaw_deg):
    """테이블 local XY의 홀 중심을 world XY로 변환한다."""
    x, y = local_xy
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return (
        origin_xy[0] + c * x - s * y,
        origin_xy[1] + s * x + c * y,
    )


def set_actor_contact_properties(actor_handle, friction):
    """Actor의 모든 collision shape에 동일한 초기 contact material을 적용한다."""
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for shape_prop in shape_props:
        shape_prop.friction = friction
        shape_prop.restitution = CONTACT_RESTITUTION
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)


def set_body_contact_properties(actor_handle, body_name, friction):
    """통합 robot actor 중 지정 rigid body의 collision shape만 설정한다."""
    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    if body_name not in body_names:
        raise RuntimeError(f"rigid body not found: {body_name}")
    body_index = body_names.index(body_name)
    shape_range = gym.get_actor_rigid_body_shape_indices(env, actor_handle)[body_index]
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for shape_index in range(shape_range.start, shape_range.start + shape_range.count):
        shape_props[shape_index].friction = friction
        shape_props[shape_index].restitution = CONTACT_RESTITUTION
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)

# ============================================================ [1] 시뮬
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="A0509 keyboard JSC/TSC/OSC")
sp = gymapi.SimParams()
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sp.dt = 1.0 / 60.0
sp.physx.solver_type = 1
sp.physx.use_gpu = True
sp.physx.num_position_iterations = 8
sp.physx.num_velocity_iterations = 1
sp.physx.contact_offset = CONTACT_OFFSET
sp.physx.rest_offset = REST_OFFSET
sp.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, pp)

# ============================================================ [2] 씬
env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0), gymapi.Vec3(1.5, 1.8, 2.2), 1)

stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True
stand_asset = gym.load_asset(sim, asset_root, A0509_STAND_URDF, stand_opts)
gym.create_actor(
    env,
    stand_asset,
    gymapi.Transform(p=gymapi.Vec3(0, 0, 0)),
    "a0509_stand",
    0,
    0,
)

# 도면 기준 고정 설비. +Y를 벽/Bonitkit 방향으로 두고, 두 테이블은
# 중앙 로봇을 감싸되 스탠드와 겹치지 않도록 0.15 m 띄운다.
fixed_opts = gymapi.AssetOptions(); fixed_opts.fix_base_link = True
bonitkit_asset = gym.load_asset(sim, asset_root, "urdf/bonitkit/bonitkit.urdf", fixed_opts)

# Table URDF가 STEP 상판에서 만든 convex prism을 collision별로 명시한다.
# V-HACD를 다시 적용하면 홀/진입 슬롯이 근사 hull로 막힐 수 있으므로 사용하지 않는다.
table_opts = gymapi.AssetOptions()
table_opts.fix_base_link = True
complete_table_asset = gym.load_asset(
    sim, asset_root, COMPLETE_TABLE_URDF, table_opts
)
prepare_table_asset = gym.load_asset(
    sim, asset_root, PREPARE_TABLE_URDF, table_opts
)

# 설비는 fixed로 유지하지만, bowl은 실제 접촉/중력에 반응하는 dynamic body다.
bowl_opts = gymapi.AssetOptions()
bowl_opts.fix_base_link = False
bowl_opts.disable_gravity = False
bowl_asset = gym.load_asset(sim, asset_root, BOWL_URDF, bowl_opts)

gym.create_actor(env, bonitkit_asset, pose(*BONITKIT_POS), "bonitkit", 0, 0)
complete_table_handle = gym.create_actor(
    env,
    complete_table_asset,
    pose(*COMPLETE_TABLE_POS, yaw_deg=TABLE_YAW_DEG),
    "complete_table",
    0,
    0,
)
prepare_table_handle = gym.create_actor(
    env,
    prepare_table_asset,
    pose(*PREPARE_TABLE_POS, yaw_deg=TABLE_YAW_DEG),
    "prepare_table",
    0,
    0,
)

# 조리 테이블의 큰 홀 1개: 원본 Ø250 mm bowl을 홀 바닥(z=0.81)에 안착.
complete_bowl_xy = local_xy_to_world(
    COMPLETE_BOWL_LOCAL_XY, COMPLETE_TABLE_POS[:2], TABLE_YAW_DEG
)
cook_bowl_handle = gym.create_actor(
    env,
    bowl_asset,
    pose(*complete_bowl_xy, COOK_BOWL_Z, yaw_deg=TABLE_YAW_DEG),
    "stirfry_bowl_cook",
    0,
    0,
)
set_actor_contact_properties(cook_bowl_handle, BOWL_FRICTION)

# 준비 테이블의 Ø200 mm 홀 10개. 실제 local 중심을 유지하고 bowl만
# 0.75배로 줄여 림 사이에 12.5 mm 간격을 둔다.
for index, local_xy in enumerate(PREPARE_BOWL_LOCAL_XY, start=1):
    bowl_xy = local_xy_to_world(local_xy, PREPARE_TABLE_POS[:2], TABLE_YAW_DEG)
    bowl_handle = gym.create_actor(
        env,
        bowl_asset,
        pose(*bowl_xy, INGREDIENT_BOWL_Z, yaw_deg=TABLE_YAW_DEG),
        f"stirfry_bowl_ingredient_{index:02d}",
        0,
        0,
    )
    gym.set_actor_scale(env, bowl_handle, INGREDIENT_BOWL_SCALE)
    set_actor_contact_properties(bowl_handle, BOWL_FRICTION)

set_actor_contact_properties(complete_table_handle, TABLE_FRICTION)
set_actor_contact_properties(prepare_table_handle, TABLE_FRICTION)

# link_6에 rigid gripper가 fixed joint로 결합된 physics asset을 사용한다.
# IK/FK는 기존 순수 A0509를 유지하며 controller 기능 자체는 변경하지 않는다.
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf=A0509_GRIPPER_URDF,
    ik_urdf=A0509_URDF,
    ee_link="link_6",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0, 0, BASE_Z)),
)
set_body_contact_properties(arm.actor, GRIPPER_BODY_NAME, GRIPPER_FRICTION)

comp_opts = gymapi.AssetOptions(); comp_opts.fix_base_link = True
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.02)), "air_compressor", 0, 0)

# ============================================================ [3] 동역학 텐서(OSC)
gym.prepare_sim(sim)
arm.setup_osc()

print(f"""[GRASP PHYSICS READY]
bowl dynamic: {not bowl_opts.fix_base_link}
bowl gravity: {not bowl_opts.disable_gravity}
bowl collision: {BOWL_COLLISION_SHAPES} explicit convex meshes
gripper rigid: True (fixed to A0509 link_6)
gripper collision: {GRIPPER_COLLISION_SHAPES} explicit convex meshes
table fixed: {table_opts.fix_base_link}
table collision: explicit convex meshes (complete={COMPLETE_TABLE_COLLISION_SHAPES}, prepare={PREPARE_TABLE_COLLISION_SHAPES})
robot grasp control: NOT TESTED""")

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(
    viewer,
    env,
    gymapi.Vec3(2.4, -2.8, 2.4),
    gymapi.Vec3(0.0, 0.25, 0.80),
)

keymap = {
    gymapi.KEY_1: "mode_jsc", gymapi.KEY_2: "mode_tsc", gymapi.KEY_3: "mode_osc",
    gymapi.KEY_W: "x+", gymapi.KEY_S: "x-",
    gymapi.KEY_A: "y+", gymapi.KEY_D: "y-",
    gymapi.KEY_Q: "z+", gymapi.KEY_E: "z-",
    gymapi.KEY_J: "joint_prev", gymapi.KEY_L: "joint_next",
    gymapi.KEY_U: "joint+",     gymapi.KEY_O: "joint-",
    gymapi.KEY_R: "home",
}
for key, act in keymap.items():
    gym.subscribe_viewer_keyboard_event(viewer, key, act)

print(f"""
========== 두산 A0509 키보드 제어 ==========
[씬] A0509 stand + {TABLE_ASSET_VERSION.upper()} tables + bowl 11개
 [모드]  1:JSC(관절)   2:TSC(좌표+IK)   3:OSC(좌표+동역학)
 [좌표 · TSC/OSC]  W/S:X±  A/D:Y±  Q/E:Z±
 [관절 · JSC]      J/L:관절선택   U/O:각도±
 [공통]  R:홈자세    (창 닫기: 종료)
=============================================""")

# ============================================================ [5] 상태 & 루프
mode = "tsc"
cart_target = np.array([0.35, 0.0, 0.55], dtype=np.float32)   # 로봇 베이스 기준
joint_target = HOME_Q.copy()
sel_joint = 0
tsc_dirty = jsc_dirty = True

def sync_cart():
    """cart_target를 현재 손끝 위치로 맞춤(모드 전환 시 튐 방지)."""
    global cart_target
    cart_target = np.array(arm.current_tcp(), dtype=np.float32)

# 시작 자세로
arm.go_joints(HOME_Q)
for _ in range(30):
    gym.simulate(sim); gym.fetch_results(sim, True)

step = 0
while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value <= 0:
            continue
        a = e.action
        if a == "mode_jsc":
            mode = "jsc"; joint_target = arm.current_joints().copy(); jsc_dirty = True; print("[모드] JSC")
        elif a == "mode_tsc":
            mode = "tsc"; sync_cart(); tsc_dirty = True; print("[모드] TSC")
        elif a == "mode_osc":
            mode = "osc"; sync_cart(); print("[모드] OSC")
        elif a in ("x+", "x-", "y+", "y-", "z+", "z-"):
            ax = {"x": 0, "y": 1, "z": 2}[a[0]]
            cart_target[ax] += (CART_STEP if a[1] == "+" else -CART_STEP)
            tsc_dirty = True
        elif a == "joint_prev":
            sel_joint = (sel_joint - 1) % 6; print(f"[관절 선택] joint_{sel_joint+1}")
        elif a == "joint_next":
            sel_joint = (sel_joint + 1) % 6; print(f"[관절 선택] joint_{sel_joint+1}")
        elif a == "joint+":
            joint_target[sel_joint] += JOINT_STEP; jsc_dirty = True
        elif a == "joint-":
            joint_target[sel_joint] -= JOINT_STEP; jsc_dirty = True
        elif a == "home":
            mode = "jsc"; joint_target = HOME_Q.copy(); jsc_dirty = True; print("[홈 자세] → JSC")

    # 선택된 모드로 제어
    if mode == "jsc":
        if jsc_dirty:
            arm.go_joints(joint_target); jsc_dirty = False
    elif mode == "tsc":
        if tsc_dirty:
            arm.go_cartesian(cart_target); tsc_dirty = False
    elif mode == "osc":
        # OSC는 월드 텐서 기준 → 장착높이 보정한 목표를 매 프레임 인가
        arm.osc_update(cart_target + np.array([0, 0, BASE_Z], dtype=np.float32))

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    if step % 60 == 0:
        tcp = arm.current_tcp()
        extra = f"  [선택 joint_{sel_joint+1}]" if mode == "jsc" else ""
        print(f"[{mode.upper()}] 목표(베이스)={np.round(cart_target,3)} 손끝={np.round(tcp,3)}{extra}")
    step += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
