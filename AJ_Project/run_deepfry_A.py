"""
run.py — 두산 A0509 키보드 다중모드 제어 (JSC / TSC / OSC)

씬:
- 하얀 바닥 2140 x 2400 mm
- 로봇 스탠드
- 두산 A0509
- 튀김기
- 오른쪽 작업대 = 준비 칸
- 왼쪽 작업대
- 바스켓 2개 = 오른쪽 준비 칸 위에 배치
- 프라이 스탠드 = 준비다이(오른쪽) / 완료다이(왼쪽) 각 1개
- 에어 컴프레셔

제어:
1 : JSC
2 : TSC
3 : OSC

TSC/OSC:
W/S : X +/-
A/D : Y +/-
Q/E : Z +/-

JSC:
J/L : 관절 선택
U/O : 관절 +/-

공통:
R : 홈 자세
"""

import os
import sys
import numpy as np

from isaacgym import gymapi, gymutil


# ============================================================
# 컨트롤러 import
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "controllers"
    )
)

from doosan_controller import DoosanController


# ============================================================
# 기본 설정
# ============================================================

from asset_config import get_asset_root
asset_root = get_asset_root()   # 컴퓨터마다 에셋 위치 자동 탐색/저장 (asset_config.py 참고)

BASE_Z = 0.81   # a0509_stand 상판(로봇 장착면) 높이

CART_STEP = 0.01       # 10 mm
JOINT_STEP = 0.05      # rad

HOME_Q = np.array(
    [0.0, 0.0, 1.2, 0.0, 1.0, 0.0],
    dtype=np.float32
)


# ============================================================
# [1] 시뮬레이터
# ============================================================

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments(
    description="A0509 keyboard JSC/TSC/OSC"
)

sp = gymapi.SimParams()

sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sp.dt = 1.0 / 60.0

sp.physx.solver_type = 1
sp.physx.use_gpu = True

sp.physx.num_position_iterations = 8
sp.physx.num_velocity_iterations = 1

sp.use_gpu_pipeline = False


sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    args.physics_engine,
    sp
)


# ============================================================
# 기본 무한 바닥
# ============================================================

pp = gymapi.PlaneParams()
pp.normal = gymapi.Vec3(0, 0, 1)

gym.add_ground(sim, pp)


# ============================================================
# [2] 환경
# ============================================================

env = gym.create_env(
    sim,
    gymapi.Vec3(-1.5, -1.5, 0),
    gymapi.Vec3(1.5, 1.5, 2),
    1
)


# ============================================================
# 하얀 바닥판
#
# X = 2140 mm
# Y = 2400 mm
# ============================================================

FLOOR_W = 2.140
FLOOR_D = 2.400
FLOOR_T = 0.02

FLOOR_Z_OFFSET = 0.002


floor_opts = gymapi.AssetOptions()
floor_opts.fix_base_link = True

floor_asset = gym.create_box(
    sim,
    FLOOR_W,
    FLOOR_D,
    FLOOR_T,
    floor_opts
)

floor_pose = gymapi.Transform()

floor_pose.p = gymapi.Vec3(
    0,
    0,
    FLOOR_Z_OFFSET - FLOOR_T / 2
)

floor_actor = gym.create_actor(
    env,
    floor_asset,
    floor_pose,
    "floor",
    0,
    0
)

gym.set_rigid_body_color(
    env,
    floor_actor,
    0,
    gymapi.MESH_VISUAL_AND_COLLISION,
    gymapi.Vec3(1.0, 1.0, 1.0)
)


# ============================================================
# 로봇 스탠드
#
# a0509_stand.urdf 원점은 바닥면(z=0) 중심이라
# 스폰 z를 0으로 주면 상판이 BASE_Z 높이에 온다.
# ============================================================

stand_opts = gymapi.AssetOptions()
stand_opts.fix_base_link = True

stand_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/a0509_stand/a0509_stand.urdf",
    stand_opts
)

stand_pose = gymapi.Transform()
stand_pose.p = gymapi.Vec3(0, 0, 0)

gym.create_actor(
    env,
    stand_asset,
    stand_pose,
    "stand",
    0,
    0
)


# ============================================================
# 작업대 에셋
# worktable2 = 약 0.9 x 0.6 x 0.95 m
# ============================================================

table_opts = gymapi.AssetOptions()
table_opts.fix_base_link = True

table_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/worktable2/worktable2.urdf",
    table_opts
)


# ============================================================
# 방향 정의
#
# 로봇 기준
#
#       뒤 (-X)
#
# 왼쪽(+Y)   로봇   오른쪽(-Y)
#
#       앞 (+X)
# ============================================================


# ============================================================
# 오른쪽 작업대 = 준비 칸
# ============================================================

RIGHT_ROT = gymapi.Quat.from_axis_angle(
    gymapi.Vec3(0, 0, 1),
    np.pi
)

# 작업대의 기하학적 중심
right_cx = 0.0

right_cy = -(
    FLOOR_D / 2 - 0.3
)

# FLOOR_D = 2.4 이므로
#
# right_cy = -(1.2 - 0.3)
#          = -0.9
#
# 즉 준비칸 중심:
# X = 0
# Y = -0.9


right_table_pose = gymapi.Transform()

right_table_pose.p = gymapi.Vec3(
    right_cx + 0.45,
    right_cy + 0.30,
    0.0
)

right_table_pose.r = RIGHT_ROT


gym.create_actor(
    env,
    table_asset,
    right_table_pose,
    "worktable2_right",
    0,
    0
)


# ============================================================
# 왼쪽 작업대
# ============================================================

left_cx = 0.0
left_cy = FLOOR_D / 2 - 0.6

left_table_pose = gymapi.Transform()

left_table_pose.p = gymapi.Vec3(
    left_cx - 0.45,
    left_cy,
    0.0
)

gym.create_actor(
    env,
    table_asset,
    left_table_pose,
    "worktable2_left",
    0,
    0
)


# ============================================================
# 튀김기
#
# 실제 크기 약:
# 0.9 x 0.6 x 1.0 m
#
# 90도 회전해서 앞쪽(+X)에 배치
# ============================================================

fryer_opts = gymapi.AssetOptions()
fryer_opts.fix_base_link = True

fryer_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/deep_fryer/deep_fryer.urdf",
    fryer_opts
)


fryer_cx = FLOOR_W / 2 - 0.3
fryer_cy = 0.0


FRYER_ROT = gymapi.Quat.from_axis_angle(
    gymapi.Vec3(0, 0, 1),
    -np.pi / 2
)


fryer_p = gymapi.Vec3(
    fryer_cx - 0.3,
    fryer_cy + 0.45,
    FLOOR_Z_OFFSET
)


fryer_pose = gymapi.Transform()
fryer_pose.p = fryer_p
fryer_pose.r = FRYER_ROT


gym.create_actor(
    env,
    fryer_asset,
    fryer_pose,
    "deep_fryer",
    0,
    0
)


# ============================================================
# 프라이어 바스켓
# ============================================================

basket_opts = gymapi.AssetOptions()

# 바스켓은 로봇이 집어야 하므로
# fix_base_link = False
#basket_opts.fix_base_link = True

basket_opts.vhacd_enabled = True

basket_opts.vhacd_params = gymapi.VhacdParams()
basket_opts.vhacd_params.resolution = 300000


basket_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/fryer_basket/fryer_basket.urdf",
    basket_opts
)


# ------------------------------------------------------------
# 바스켓 실제 STL 기준
#
# x = [-0.1947, 0.4116]
# y = [-0.0855, 0.0855]
# z = [-0.0062, 0.2558]
#
# 바스켓 형상의 중심과 URDF 원점이 일치하지 않음
# ------------------------------------------------------------

BASKET_X_OFF = 0.10846
BASKET_Z_MIN = -0.0062
BASKET_W = 0.171


# 손잡이가 로봇 쪽을 바라보게 180도 회전
BASKET_HANDLE_ROT = gymapi.Quat.from_axis_angle(
    gymapi.Vec3(0, 0, 1),
    -np.pi / 2
)


# ============================================================
# ★ 준비 칸 바스켓 배치
#
# 오른쪽 작업대 중심
#
# X = right_cx = 0
# Y = right_cy = -0.9
#
# 작업대 높이 = 0.95 m 실제위치 = 0.80
# ============================================================

READY_X = left_cx
READY_Y = left_cy
READY_Z = 0.95


NUM_BASKETS = 4

# 바스켓 폭이 약 171 mm이므로
# 200 mm 간격으로 나란히 배치
BASKET_SPACING = 0.20


for i in range(NUM_BASKETS):

    # Y 방향으로 두 개 나란히
    x_off = (
        i - (NUM_BASKETS - 1) / 2
    ) * BASKET_SPACING


    # 180도 회전 시
    # 바스켓 형상 중심이 원점 기준 -X 쪽으로 이동하므로
    # actor 원점을 +BASKET_X_OFF 해준다.
    basket_pose = gymapi.Transform()

    basket_pose.p = gymapi.Vec3(
        READY_X + BASKET_X_OFF + x_off,
        READY_Y + 0.25,
        READY_Z - BASKET_Z_MIN
    )

    basket_pose.r = BASKET_HANDLE_ROT


    gym.create_actor(
        env,
        basket_asset,
        basket_pose,
        f"fryer_basket_{i}",
        0,
        0
    )


# ============================================================
# 프라이 스탠드 (준비/완료 거치대)
#
# fry_stand.urdf: 준비/완료 두 구역이 한 몸체로 합쳐진 거치대.
# 원점이 바닥면 중심 부근(x:-0.4525~0.4525, y:-0.28~0.323)이라
# 별도 코너 보정 없이 테이블의 실제 기하학적 중심 좌표에 그대로 얹는다.
#
# 준비다이 = 오른쪽 작업대(right_cx, right_cy)
# 완료다이 = 왼쪽 작업대. 단, 왼쪽 작업대는 배치 시 y 코너 보정(+0.30)이
# 빠져 있어 실제 중심이 (left_cx, left_cy + 0.30)에 있음 — 그 실제 위치에 맞춘다.
# ============================================================

stand_opts2 = gymapi.AssetOptions()
stand_opts2.fix_base_link = True
stand_opts2.vhacd_enabled = True
stand_opts2.vhacd_params = gymapi.VhacdParams()
stand_opts2.vhacd_params.resolution = 300000

fry_stand_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/fry_stand/fry_stand.urdf",
    stand_opts2
)


# worktable2 STL을 법선+면적 기준으로 분석한 실측값(전체 bbox의 z_max=0.95는
# 뒷판(backsplash) 상단 얇은 띠(0.9x0.04, 면적 0.036m^2)일 뿐, 실제 상판은
# z=0.85(면적 0.504m^2, 발자국 대부분)이다. fry_stand 로컬 최저점(z=0.0)을
# 이 실제 상판에 맞춘다.
TABLE_TOP_Z = 0.85

# 준비다이 위 (오른쪽 작업대)
ready_stand_pose = gymapi.Transform()
ready_stand_pose.p = gymapi.Vec3(right_cx, right_cy, TABLE_TOP_Z)
ready_stand_pose.r = RIGHT_ROT

gym.create_actor(
    env,
    fry_stand_asset,
    ready_stand_pose,
    "fry_stand_ready",
    0,
    0
)

# 완료다이 위 (왼쪽 작업대, 실제 중심 y = left_cy + 0.30)
done_stand_pose = gymapi.Transform()
done_stand_pose.p = gymapi.Vec3(left_cx, left_cy + 0.30, TABLE_TOP_Z)

gym.create_actor(
    env,
    fry_stand_asset,
    done_stand_pose,
    "fry_stand_done",
    0,
    0
)


# ============================================================
# 두산 A0509
#
# 스탠드 상단 Z = 0.8 m
# ============================================================

arm = DoosanController(
    gym,
    sim,
    env,
    asset_root,

    urdf="urdf/doosan_a0509/a0509.urdf",

    fix_base=True,

    spawn_transform=gymapi.Transform(
        p=gymapi.Vec3(
            0,
            0,
            BASE_Z
        )
    ),
)


# ============================================================
# 에어 컴프레셔
# ============================================================

comp_opts = gymapi.AssetOptions()
comp_opts.fix_base_link = True


comp_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/air_compressor/air_compressor.urdf",
    comp_opts
)


comp_pose = gymapi.Transform()

comp_pose.p = gymapi.Vec3(
    0,
    0,
    0.02
)


gym.create_actor(
    env,
    comp_asset,
    comp_pose,
    "air_compressor",
    0,
    0
)


# ============================================================
# [3] OSC용 동역학 텐서
# ============================================================

gym.prepare_sim(sim)

arm.setup_osc()


# ============================================================
# [4] Viewer
# ============================================================

viewer = gym.create_viewer(
    sim,
    gymapi.CameraProperties()
)


gym.viewer_camera_look_at(
    viewer,
    env,

    gymapi.Vec3(
        1.6,
        1.6,
        1.6
    ),

    gymapi.Vec3(
        0,
        0,
        0.9
    )
)


# ============================================================
# 키 등록
# ============================================================

# 키보드 텔레오프 (JSC/TSC/OSC, 꾹 누르면 연속, TSC 자세 유지) — controllers/doosan_arm_keyboard_teleop.py
from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop
teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)

step = 0
while not gym.query_viewer_has_closed(viewer):
    teleop.handle_and_apply()   # 이벤트 처리 + 연속동작 + 모드별 제어
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
    step += 1

gym.destroy_viewer(viewer)

gym.destroy_sim(sim)

print("종료.")

