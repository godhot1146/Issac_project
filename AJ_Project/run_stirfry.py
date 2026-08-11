"""
run_stirfry.py — 두산 A0509 볶음 공정 씬 + 수동/자동 제어

씬: 볶음 도면을 기준으로 Bonitkit + V2 조리/준비 테이블 + 그릇 11개를
    A0509 작업 반경 안에 배치한다. A0509는 전용 stand 상단 z=0.81에 고정 장착한다.
제어: 기본은 실행 중 키로 모드를 바꿔가며 직접 조작한다.
      --auto를 주면 그리퍼를 조리 그릇 림보다 10 mm 위까지 자동으로
      이동한 뒤, 현재 자세를 유지한 채 키보드 미세조작으로 전환한다.

  ┌─────────────── 키 맵 ───────────────┐
  │ [모드]  1: JSC(관절)  2: TSC(좌표+IK)  3: OSC(좌표+동역학)
  │ [좌표이동 · TSC/OSC]  W/S: X±   A/D: Y±   Q/E: Z±
  │ [자세회전 · TSC]      T/G: pitch±  F/H: roll±  C/V: yaw±
  │ [관절이동 · JSC]      J/L: 관절 선택   U/O: 선택관절 각도±
  │ [공통]  R: 홈(전 관절 0, 천장 방향)   (뷰어 창 닫기: 종료)
  └──────────────────────────────────────┘

제어 로직은 controllers/doosan_controller.py와
controllers/doosan_arm_keyboard_teleop.py에 분리되어 있다.
좌표 목표는 '로봇 베이스 기준'. OSC는 월드 텐서를 쓰므로 장착높이(+0.81) 보정해 넘긴다.

실행:  conda activate issac_env  &&  python run_stirfry.py
       python run_stirfry.py --auto   # 자동 접근 -> 키보드 미세조작
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
args = gymutil.parse_arguments(
    description="A0509 manual teleop / automatic bowl grasp and pour",
    custom_parameters=[
        {
            "name": "--auto",
            "action": "store_true",
            "help": "그릇 앞까지 자동 접근한 뒤 키보드 미세조작으로 전환한다.",
        }
    ],
)
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
robot grasp control: {"HYBRID (auto approach -> manual)" if args.auto else "MANUAL"}""")

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(
    viewer,
    env,
    gymapi.Vec3(2.4, -2.8, 2.4),
    gymapi.Vec3(0.0, 0.25, 0.80),
)

from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop

# --auto는 접촉 직전의 안전 위치까지만 자동 이동한다. 파지·상승·붓기는
# 실행하지 않고, 도착한 현재 자세를 그대로 TSC 키보드 조작기에 넘긴다.
if args.auto:
    from stirfry_auto_sequence import StirfryAutoSequence

    auto_sequence = StirfryAutoSequence(
        gym,
        sim,
        env,
        arm,
        cook_bowl_handle,
        base_z=BASE_Z,
        dt=sp.dt,
        slot_direction_xy=(-1.0, 0.0),
        manual_handoff=True,
    )
    teleop = None
else:
    # 키보드 텔레오프 (JSC/TSC/OSC, 꾹 누르면 연속, TSC 자세 유지)
    # 구현: controllers/doosan_arm_keyboard_teleop.py
    teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)
    auto_sequence = None

while not gym.query_viewer_has_closed(viewer):
    if auto_sequence is not None:
        auto_sequence.update()
        if auto_sequence.handoff_ready:
            # 일반 수동 모드의 4 mm/프레임보다 느린 1 mm/프레임과
            # 0.5 deg/프레임을 사용해 림 주변 접촉을 미세 조정한다.
            teleop = DoosanArmKeyboardTeleop(
                gym,
                viewer,
                arm,
                base_z=BASE_Z,
                cart_step=0.001,
                joint_step=np.deg2rad(0.2),
                ori_step_deg=0.5,
                settle=0,
                home_on_start=False,
            )
            auto_sequence = None
            print("""
===== 자동 접근 -> 키보드 미세조작 전환 완료 =====
 1) 2번: TSC 재동기화(권장)
 2) W/S: X±, A/D: Y±, Q/E: Z± (1 mm/프레임)
 3) T/G: pitch±, F/H: roll±, C/V: yaw± (0.5 deg/프레임)
 4) 필요하면 1번 JSC -> J/L 관절 선택 -> U/O 미세 회전
 주의: R은 홈 복귀이므로 파지 중에는 누르지 마세요.
==================================================""")
    else:
        teleop.handle_and_apply()   # 이벤트 처리 + 연속동작 + 모드별 제어

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
