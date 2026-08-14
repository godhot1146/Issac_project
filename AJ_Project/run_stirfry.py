"""
run_stirfry.py — 두산 A0509 볶음 공정 씬 + 수동/자동 제어

씬: 볶음 수정 도면을 기준으로 Bonitkit + 4구 테이블 3대 + 그릇 12개를
    A0509 주위의 서/북/남 3면에 배치한다. 캐비닛 긴 축은 Bonitkit을 향하고,
    서쪽 테이블 아래로 캐비닛이 일부 들어간다. 아직 상판에 홀이
    없으므로 그릇은 향후 홀 중심이 될 위치의 상판 위에 올린다.
제어: 기본은 실행 중 키로 모드를 바꿔가며 직접 조작한다.
      --auto를 주면 그리퍼를 조리 그릇 림 접촉 기준보다 180 mm 위까지
      이동한 뒤, 현재 자세를 유지한 채 키보드 미세조작으로 전환한다.
      --auto-grasp를 주면 같은 안전 접근 뒤 상부 고리 접촉점을 축으로
      팔 전체를 연속 이동해 자동 파지·상승한다.

  ┌─────────────── 키 맵 ───────────────┐
  │ [모드]  1: JSC(관절)  2: TSC(좌표+IK)  3: OSC(좌표+동역학)
  │ [좌표이동 · TSC/OSC]  W/S: X±   A/D: Y±   Q/E: Z±
  │ [자세회전 · TSC]      T/G: pitch±  F/H: roll±  C/V: yaw±
  │ [관절이동 · JSC]      J/L: 관절 선택   U/O: 선택관절 각도±
  │ [공통]  R: 홈(전 관절 0, 천장 방향)   (뷰어 창 닫기: 종료)
  └──────────────────────────────────────┘

제어 로직은 controllers/doosan_controller.py와
controllers/doosan_arm_keyboard_teleop.py에 분리되어 있다.
좌표 목표는 '로봇 베이스 기준'. OSC는 월드 텐서를 쓰므로 장착높이(+0.805) 보정해 넘긴다.

실행:  conda activate issac_env  &&  python run_stirfry.py
       python run_stirfry.py --auto   # 자동 접근 -> 키보드 미세조작
       python run_stirfry.py --auto-grasp  # 접촉점 피벗 자동 파지 -> 상승
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

CABINET_HEIGHT = 0.805
BASE_Z = CABINET_HEIGHT

# 로봇 베이스는 world (0, 0), Bonitkit 정면은 +X다. 0.60 x 0.90 m
# 캐비닛을 +90 deg 회전해 긴 축을 X에 맞추고, 로봇을 +X 끝에서 150 mm
# 안쪽에 둔다. 즉 캐비닛 중심은 로봇보다 -X로 300 mm 떨어져 있다.
ROBOT_CABINET_POS = (-0.300, 0.000, 0.0)
ROBOT_CABINET_YAW_DEG = 90.0

# 통로를 없애고 Bonitkit 개방면이 서쪽(-X)의 로봇을 향하도록 -90 deg
# 회전한다. 회전 뒤 X 방향 폭은 0.935 m이며 메쉬 원점의 y 오프셋까지 보정한
# actor 원점 x=1.0615 m를 사용한다. 개방면의 서쪽 끝은 로봇에서 0.5865 m,
# side table과의 X 간격은 11.5 mm다.
BONITKIT_POS = (1.0615, 0.000, 0.0)
BONITKIT_YAW_DEG = -90.0

# 북/서/남 모두 같은 0.30 x 1.15 x 0.90 m 테이블을 사용한다.
# 세 테이블 중심은 로봇에서 같은 0.735 m 반경에 둬 대응 그릇 거리를
# 동일하게 유지한다. 테이블 모서리 간격은 10 mm, 서쪽 테이블 아래로
# 들어가는 캐비닛 길이는 165 mm이며 다리는 캐비닛 양옆으로 245 mm 떨어진다.
TABLE_LAYOUT = (
    ("west",  (-0.735,  0.000, 0.0),   0.0),
    ("north", ( 0.000,  0.735, 0.0),  90.0),
    ("south", ( 0.000, -0.735, 0.0), -90.0),
)
TABLE_TOP_Z = 0.900
BOWL_SPAWN_CLEARANCE = 0.001
ROBOT_CABINET_URDF  = "urdf/robot_cabinetnplate/robot_cabinetnplate.urdf"
AIR_COMPRESSOR_URDF = "urdf/air_compressor/air_compressor.urdf"
DOOSAN_CONTROLLER_URDF = "urdf/doosan_controller/doosan_controller.urdf"
STIRFRY_STAGE_URDF   = "urdf/stirfry_stage_300x1150/stirfry_stage_300x1150.urdf"
BOWL_URDF            = "urdf/stirfry_bowl/stirfry_bowl.urdf"
A0509_URDF           = "urdf/doosan_a0509/a0509.urdf"
A0509_GRIPPER_URDF   = "urdf/a0509_stirfry_gripper/a0509_stirfry_gripper.urdf"
GRIPPER_BODY_NAME    = "stirfry_gripper_link"

# 캐비닛 메쉬는 내부가 보이지만 collision은 전체 박스 하나로 단순화돼 있다.
# 동일 bit를 준 내부 고정 설비와 캐비닛 사이의 가짜 접촉만 거르고,
# 로봇(filter=1) 및 다른 설비(filter=0)와의 충돌은 유지한다.
CABINET_INTERNAL_COLLISION_FILTER = 2
AIR_COMPRESSOR_LOCAL_POS = (-0.2293, -0.1591, 0.1960)
DOOSAN_CONTROLLER_LOCAL_POS = (-0.2017, 0.2472, 0.1090)

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
STIRFRY_STAGE_COLLISION_SHAPES = 5  # 상판 1 + 다리 4

# 세 테이블 모두 같은 Ø250 mm 그릇 4개와 같은 250 mm pitch를 사용한다.
TABLE_BOWL_LOCAL_XY = tuple((0.0, s) for s in (-0.375, -0.125, 0.125, 0.375))


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


def local_xyz_pose(local_xyz, origin_xyz, yaw_deg):
    """설비 local XYZ를 회전된 부모 설비의 world pose로 변환한다."""
    world_x, world_y = local_xy_to_world(local_xyz[:2], origin_xyz[:2], yaw_deg)
    return pose(
        world_x,
        world_y,
        origin_xyz[2] + local_xyz[2],
        yaw_deg=yaw_deg,
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
        },
        {
            "name": "--auto-grasp",
            "action": "store_true",
            "help": "상부 고리 접촉점 피벗 방식의 자동 그릇 파지·상승을 실행한다.",
        }
    ],
)
if args.auto and args.auto_grasp:
    raise ValueError("--auto와 --auto-grasp는 동시에 사용할 수 없습니다.")
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
env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0), gymapi.Vec3(2.6, 1.5, 2.2), 1)

fixture_opts = gymapi.AssetOptions(); fixture_opts.fix_base_link = True
cabinet_asset = gym.load_asset(sim, asset_root, ROBOT_CABINET_URDF, fixture_opts)
gym.create_actor(
    env,
    cabinet_asset,
    pose(*ROBOT_CABINET_POS, yaw_deg=ROBOT_CABINET_YAW_DEG),
    "robot_cabinetnplate",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)

# 컴프레서와 컨트롤러도 캐비닛 local 좌표를 유지한 채 +90 deg 함께 회전한다.
compressor_asset = gym.load_asset(sim, asset_root, AIR_COMPRESSOR_URDF, fixture_opts)
controller_asset = gym.load_asset(sim, asset_root, DOOSAN_CONTROLLER_URDF, fixture_opts)
gym.create_actor(
    env,
    compressor_asset,
    local_xyz_pose(
        AIR_COMPRESSOR_LOCAL_POS,
        ROBOT_CABINET_POS,
        ROBOT_CABINET_YAW_DEG,
    ),
    "air_compressor_in_cabinet",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)
gym.create_actor(
    env,
    controller_asset,
    local_xyz_pose(
        DOOSAN_CONTROLLER_LOCAL_POS,
        ROBOT_CABINET_POS,
        ROBOT_CABINET_YAW_DEG,
    ),
    "doosan_controller_in_cabinet",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)

# 도면 기준 고정 설비. +X Bonitkit, -X/+/-Y에 같은 300x1150 테이블 3대다.
fixed_opts = gymapi.AssetOptions(); fixed_opts.fix_base_link = True
bonitkit_asset = gym.load_asset(sim, asset_root, "urdf/bonitkit/bonitkit.urdf", fixed_opts)

table_opts = gymapi.AssetOptions()
table_opts.fix_base_link = True
table_asset = gym.load_asset(sim, asset_root, STIRFRY_STAGE_URDF, table_opts)

# 설비는 fixed로 유지하지만, bowl은 실제 접촉/중력에 반응하는 dynamic body다.
bowl_opts = gymapi.AssetOptions()
bowl_opts.fix_base_link = False
bowl_opts.disable_gravity = False
bowl_asset = gym.load_asset(sim, asset_root, BOWL_URDF, bowl_opts)

gym.create_actor(
    env,
    bonitkit_asset,
    pose(*BONITKIT_POS, yaw_deg=BONITKIT_YAW_DEG),
    "bonitkit",
    0,
    0,
)
table_handles = []
bowl_handles = []
cook_bowl_handle = None

for face_name, table_pos, table_yaw_deg in TABLE_LAYOUT:
    table_handle = gym.create_actor(
        env,
        table_asset,
        pose(*table_pos, yaw_deg=table_yaw_deg),
        f"stirfry_stage_{face_name}",
        0,
        0,
    )
    table_handles.append(table_handle)
    set_actor_contact_properties(table_handle, TABLE_FRICTION)

    # 아직 홀이 없으므로 각 예정 홀 중심의 상판보다 1 mm 위에 생성한다.
    for slot_index, local_xy in enumerate(TABLE_BOWL_LOCAL_XY, start=1):
        bowl_xy = local_xy_to_world(local_xy, table_pos[:2], table_yaw_deg)
        bowl_handle = gym.create_actor(
            env,
            bowl_asset,
            pose(
                *bowl_xy,
                TABLE_TOP_Z + BOWL_SPAWN_CLEARANCE,
                yaw_deg=table_yaw_deg,
            ),
            f"stirfry_bowl_{face_name}_{slot_index:02d}",
            0,
            0,
        )
        bowl_handles.append(bowl_handle)
        set_actor_contact_properties(bowl_handle, BOWL_FRICTION)

        # 북쪽 테이블의 안쪽 그릇 하나를 대표 자동 파지 대상으로 사용한다.
        if face_name == "north" and slot_index == 2:
            cook_bowl_handle = bowl_handle

if cook_bowl_handle is None:
    raise RuntimeError("automatic grasp target bowl was not created")

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
table layout: 3 x identical 300x1150 stages, top z={TABLE_TOP_Z:.3f} m
table collision: {STIRFRY_STAGE_COLLISION_SHAPES} shapes per stage
bowl layout: {len(bowl_handles)} bowls on uncut tabletops
robot fixture: A0509 mounted directly on cabinet top z={BASE_Z:.3f} m
cabinet: long axis faces Bonitkit; robot is 0.150 m from its +X end
west underlap: cabinet 0.165 m under table; tabletop clearance=0.055 m
Bonitkit nearest face: x=0.5865 m from robot base
cabinet equipment: air compressor + Doosan controller, centered side-by-side
robot grasp control: {"RECORDED AUTO GRASP" if args.auto_grasp else ("HYBRID (auto approach -> manual)" if args.auto else "MANUAL")}""")

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(
    viewer,
    env,
    gymapi.Vec3(2.4, -2.6, 2.3),
    gymapi.Vec3(0.15, 0.0, 0.85),
)

from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop
from stirfry_arm_keyboard_teleop import StirfryArmKeyboardTeleop
from stirfry_teleop_recorder import StirfryTeleopRecorder

# --auto는 림 접촉 기준보다 180 mm 위의 안전 위치까지만 자동 이동한다.
# 파지·상승·붓기는 실행하지 않고, 현재 자세를 TSC 키보드 조작기에 넘긴다.
if args.auto_grasp:
    from stirfry_recorded_grasp_sequence import StirfryRecordedGraspSequence

    auto_sequence = StirfryRecordedGraspSequence(
        gym,
        sim,
        env,
        arm,
        cook_bowl_handle,
        base_z=BASE_Z,
        dt=sp.dt,
        slot_direction_xy=(0.0, 1.0),
    )
    teleop = None
    teleop_recorder = None
elif args.auto:
    from stirfry_auto_sequence import StirfryAutoSequence

    auto_sequence = StirfryAutoSequence(
        gym,
        sim,
        env,
        arm,
        cook_bowl_handle,
        base_z=BASE_Z,
        dt=sp.dt,
        slot_direction_xy=(0.0, 1.0),
        manual_handoff=True,
    )
    teleop = None
    teleop_recorder = None
else:
    # 키보드 텔레오프 (JSC/TSC/OSC, 꾹 누르면 연속, TSC 자세 유지)
    # 구현: controllers/doosan_arm_keyboard_teleop.py
    teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)
    auto_sequence = None
    teleop_recorder = None

while not gym.query_viewer_has_closed(viewer):
    success_requested = False
    if auto_sequence is not None:
        auto_sequence.update()
        if auto_sequence.handoff_ready:
            gravity_control = auto_sequence.auto_control
            # 일반 수동 모드의 4 mm/프레임보다 느린 1 mm/프레임과
            # 0.5 deg/프레임을 사용해 림 주변 접촉을 미세 조정한다.
            teleop = StirfryArmKeyboardTeleop(
                gym,
                viewer,
                arm,
                base_z=BASE_Z,
                cart_step=0.001,
                joint_step=np.deg2rad(0.2),
                ori_step_deg=0.5,
                settle=0,
                home_on_start=False,
                gravity_control=gravity_control,
            )
            auto_sequence = None
            teleop_recorder = StirfryTeleopRecorder(
                gym,
                env,
                arm,
                cook_bowl_handle,
                teleop,
                dt=sp.dt,
                gripper_body_name=GRIPPER_BODY_NAME,
            )
            print("""
===== 자동 접근 -> 키보드 미세조작 전환 완료 =====
 1) 2번: 현재 자세에서 안전 TSC 재동기화(선택)
 2) W/S: X±, A/D: Y±, Q/E: Z± (1 mm/프레임)
 3) T/G: pitch±, F/H: roll±, C/V: yaw± (0.5 deg/프레임)
 4) 필요하면 1번 JSC -> J/L 관절 선택 -> U/O 미세 회전
 5) 그릇을 들었으면 ENTER: 성공 표시 + 분석용 로그 출력
 6) 3번 OSC는 안전을 위해 비활성화
 주의: R은 홈 복귀이므로 파지 중에는 누르지 마세요.
==================================================""")
    else:
        teleop.handle_and_apply()   # 이벤트 처리 + 연속동작 + 모드별 제어
        if teleop_recorder is not None:
            success_requested = teleop.consume_success_marker()

    gym.simulate(sim); gym.fetch_results(sim, True)
    if teleop_recorder is not None and teleop_recorder.active:
        teleop_recorder.record(force=success_requested)
        if success_requested:
            teleop_recorder.finish("user_marked_success")
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

if teleop_recorder is not None and teleop_recorder.active:
    teleop_recorder.record(force=True)
    teleop_recorder.finish("viewer_closed_without_success_marker")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
