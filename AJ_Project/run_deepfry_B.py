"""
run.py — 두산 A0509 키보드 다중모드 제어 (JSC / TSC / OSC)

씬: 선반(고정) + A0509 팔(선반 상단 z=0.8에 고정 장착) + 에어 컴프레셔(선반 안).
제어: 실행 중 키로 모드를 바꿔가며 직접 조작.

  ┌─────────────── 키 맵 ───────────────┐
  │ [모드]  1: JSC(관절)  2: TSC(좌표+IK)  3: OSC(좌표+동역학)
  │ [좌표이동 · TSC/OSC]  W/S: X±   A/D: Y±   Q/E: Z±
  │ [관절이동 · JSC]      J/L: 관절 선택   U/O: 선택관절 각도±
  │ [공통]  R: 홈 자세    (뷰어 창 닫기: 종료)
  └──────────────────────────────────────┘

제어 로직은 controllers/doosan_controller.py (JSC/TSC/OSC 통합).
좌표 목표는 '로봇 베이스 기준'. OSC는 월드 텐서를 쓰므로 장착높이(+0.8) 보정해 넘긴다.

실행:  conda activate issac_env  &&  python run.py     (OSC용 gymtorch→ninja 필요)
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil   # torch보다 먼저

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers"))
from doosan_controller import DoosanController

asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")

BASE_Z     = 0.8      # 팔 장착 높이(선반 상단면)
CART_STEP  = 0.01     # 좌표 목표 이동 스텝(m)
JOINT_STEP = 0.05     # 관절 이동 스텝(rad)
HOME_Q     = np.array([0.0, 0.0, 1.2, 0.0, 1.0, 0.0], dtype=np.float32)

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
sp.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, pp)

# ============================================================ [2] 씬
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

# 로봇팔 XY 원점(0,0) 기준 하얀색 바닥판 (가로 2140mm x 세로 2400mm)
FLOOR_W = 2.140   # 가로(X), m
FLOOR_D = 2.400   # 세로(Y), m
FLOOR_T = 0.02    # 두께, m
FLOOR_Z_OFFSET = 0.002  # 기존 무한 바닥면과 겹쳐 z-fighting(깜빡임) 나지 않도록 살짝 띄움
floor_opts = gymapi.AssetOptions(); floor_opts.fix_base_link = True
floor_asset = gym.create_box(sim, FLOOR_W, FLOOR_D, FLOOR_T, floor_opts)
floor_actor = gym.create_actor(
    env, floor_asset,
    gymapi.Transform(p=gymapi.Vec3(0, 0, FLOOR_Z_OFFSET - FLOOR_T / 2)),
    "floor", 0, 0,
)
gym.set_rigid_body_color(env, floor_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 1.0, 1.0))

stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True
stand_asset = gym.load_asset(sim, asset_root, "urdf/robot_stand/robot_stand.urdf", stand_opts)
gym.create_actor(env, stand_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0)), "stand", 0, 0)

# 작업대(WorkTable2, STEP→URDF 변환) 에셋. 오른쪽/뒤 선반 대용으로 아래에서 재사용.
table_opts = gymapi.AssetOptions(); table_opts.fix_base_link = True
table_asset = gym.load_asset(sim, asset_root, "urdf/worktable2/worktable2.urdf", table_opts)

# ---- 방향 정의: 로봇 기본 리치(cart_target=(0.35,0,0.55), 베이스 기준)가 +X이므로
#      앞=+X, 뒤=-X, 오른쪽=-Y, 왼쪽=+Y 로 정의하고 아래 배치에 사용.

# 튀김기(Deep Fryer, 다운로드 폴더 STEP→URDF 변환, 0.9x0.6x1.0m). 90도 회전된 상태(발자국 x폭 0.6/y폭 0.9)라
# 바닥 앞쪽 끝(+X, x=1.07)에 딱 맞춰 정렬. y는 로봇 중심(0)에 맞춤.
fryer_opts = gymapi.AssetOptions(); fryer_opts.fix_base_link = True
# 원본 메쉬(삼각형 68개) 그대로 충돌 처리하면 바스켓과의 접촉이 불안정해서(위 상판의 평평한 두 삼각형 경계에서
# 흔들림 발생) VHACD 볼록분해로 로드 — 정적 바디라 성능 영향은 미미.
fryer_opts.vhacd_enabled = True
fryer_opts.vhacd_params = gymapi.VhacdParams()
fryer_opts.vhacd_params.resolution = 300000
fryer_asset = gym.load_asset(sim, asset_root, "urdf/deep_fryer/deep_fryer.urdf", fryer_opts)
fryer_cx = FLOOR_W / 2 - 0.3    # 회전 후 x반폭 0.3m → 바닥 앞쪽 끝(+X)에 맞춤
fryer_cy = 0.0                  # 회전 후 y반폭 0.45m, 로봇 중심(0)에 맞춤
FRYER_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)  # 위에서 볼 때 시계방향 90도
# 회전 후 로컬 원점(0,0)의 월드 위치: 중심 기준 오프셋(-0.45,-0.3)을 -90도 회전시켜 재배치
fryer_p = gymapi.Vec3(fryer_cx - 0.3, fryer_cy + 0.45, FLOOR_Z_OFFSET)
gym.create_actor(env, fryer_asset, gymapi.Transform(p=fryer_p, r=FRYER_ROT), "deep_fryer", 0, 0)

# 프라이어 바스켓(Fryer_Basket, STEP→URDF 변환) 여러 개를 튀김기 위에 나란히 배치.
# 와이어프레임 형상 → VHACD 볼록분해로 로드. 그리퍼로 집어 옮기는 대상이므로 fix_base_link=False(기본).
basket_opts = gymapi.AssetOptions()
basket_opts.vhacd_enabled = True
basket_opts.vhacd_params = gymapi.VhacdParams()
basket_opts.vhacd_params.resolution = 300000
basket_asset = gym.load_asset(sim, asset_root, "urdf/fryer_basket/fryer_basket.urdf", basket_opts)

# 실측(STL) 기준 로컬 치수: x:[-0.1947,0.4116](손잡이가 +x쪽으로 돌출, 반길이 0.303), y:[-0.0855,0.0855], z:[-0.0062,0.2558]
BASKET_X_OFF = 0.10846   # 로컬 원점→중심 x오프셋 (무회전 기준)
BASKET_HALF_LEN = 0.303  # 로컬 x 반길이 (손잡이 포함)
BASKET_Z_MIN = -0.0062   # 로컬 바닥면(원점 기준)
BASKET_W = 0.171         # 바스켓 폭(월드 Y로 나란히 배치할 때 사용)
BASKET_HANDLE_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)  # 손잡이(로컬 +X)가 로봇팔 쪽(-X)을 향하도록 180도 회전

# 튀김기 상판은 평평한 한 면이 아니라 2단 구조: 대부분(로컬 y:0~0.513, 회전 후 월드 x:fryer_p.x~+0.513)은
# 로컬 z=0.85(메인 상판)이고, 앞쪽 얇은 턱(로컬 y:0.573~0.6, 월드 x:+0.573~+0.6쪽)만 z=1.0(전체 높이)로 15cm 더 높다.
# 바스켓 길이(0.606m)가 메인 상판 폭(0.513m)보다 길어서 그 턱에 살짝 걸치면 스폰 시 겹쳐서(관통) 튕겨나간다.
# → 바스켓 열의 중심을 턱 쪽 경계에서 半길이+여유만큼 뒤로 물려서, 턱과는 안 겹치고 뒤쪽(개방된 면)으로만 살짝 걸치게 배치.
fryer_top_z = FLOOR_Z_OFFSET + 0.85
FLAT_TOP_X_MAX = fryer_cx + 0.213   # 메인 상판과 턱의 경계(월드 x)
basket_row_cx = FLAT_TOP_X_MAX - BASKET_HALF_LEN - 0.01  # 턱과 1cm 여유
NUM_BASKETS = 4
BASKET_SPACING = 0.20   # 겹치지 않게 여유(바스켓 폭 0.171m보다 약간 큼)
bottom_z = fryer_top_z  # 쌓지 않고 튀김기 상판 위에 한 줄로
for i in range(NUM_BASKETS):
    y_off = (i - (NUM_BASKETS - 1) / 2) * BASKET_SPACING
    # 180도 회전이므로 중심 기준 원점 오프셋 부호가 반대가 됨: center - BASKET_X_OFF → center + BASKET_X_OFF
    pose = gymapi.Transform(
        p=gymapi.Vec3(basket_row_cx + BASKET_X_OFF, fryer_cy + y_off, bottom_z - BASKET_Z_MIN),
        r=BASKET_HANDLE_ROT,
    )
    gym.create_actor(env, basket_asset, pose, f"fryer_basket_{i}", 0, 0)

# 선반 대용으로 worktable2를 로봇팔 오른쪽(-Y)과 뒤(-X)에 각각 1개씩(총 2개) 배치.
# worktable2(0.9x0.6x0.95m)는 cargo_shelf보다 작아 스탠드/튀김기/작업대와 겹치지 않고 바닥 안에(뒤쪽만 살짝 걸침) 들어간다.
# 오른쪽 작업대: 180도 회전(풋프린트 x반폭 0.45/y반폭 0.3) → 바닥 오른쪽 끝(-Y, y=-1.2)에 맞춤, x는 로봇 중심(0)에 맞춤.
RIGHT_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)
right_cx, right_cy = 0.0, -(FLOOR_D / 2 - 0.3)
gym.create_actor(env, table_asset, gymapi.Transform(p=gymapi.Vec3(right_cx + 0.45, right_cy + 0.3, 0.0), r=RIGHT_ROT), "worktable2_right", 0, 0)
# 왼쪽 작업대(원래 뒤쪽에 있던 것을 이동): 무회전, 바닥 왼쪽 끝(+Y, y=1.2)에 맞춤, x는 로봇 중심(0)에 맞춤.
left_cx, left_cy = 0.0, FLOOR_D / 2 - 0.6
gym.create_actor(env, table_asset, gymapi.Transform(p=gymapi.Vec3(left_cx - 0.45, left_cy, 0.0)), "worktable2_left", 0, 0)

# 순수 A0509를 선반 상단(z=0.8)에 고정 장착 (OSC 동역학이 깔끔하도록 고정베이스 순수팔)
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf="urdf/doosan_a0509/a0509.urdf",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0, 0, BASE_Z)),
)

comp_opts = gymapi.AssetOptions(); comp_opts.fix_base_link = True
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.02)), "air_compressor", 0, 0)

# 미니PC(Mini_PC, STEP→URDF 변환). 스탠드 수납공간 안, 에어컴프레셔(x:-0.23~0.23,y:-0.16~0.16) 옆 빈 자리에 배치.
pc_opts = gymapi.AssetOptions(); pc_opts.fix_base_link = True
pc_asset = gym.load_asset(sim, asset_root, "urdf/mini_pc/mini_pc.urdf", pc_opts)
gym.create_actor(env, pc_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.22, 0.02)), "mini_pc", 0, 0)

# 태블릿(Tablet, STEP→URDF 변환). 왼쪽 작업대(worktable2_left) 뒤편(월드 y:1.16~1.2, 로컬 상판보다 10cm 더 높은
# 얇은 뒷턱/기둥, 상판 y:0~0.56은 z=0.85·뒤턱만 z=0.95)의 위쪽(z=0.95)에 세워서 붙임.
# 세로(긴 변 235.7mm)가 위로 서도록 로컬 X→월드 Z, 얇은 두께(로컬 Z)→월드 -Y(로봇 쪽을 바라보게), 로컬 Y→월드 -X로 회전.
tablet_opts = gymapi.AssetOptions(); tablet_opts.fix_base_link = True
tablet_asset = gym.load_asset(sim, asset_root, "urdf/tablet/tablet.urdf", tablet_opts)
TABLET_ROT = gymapi.Quat(0.5, -0.5, 0.5, 0.5)
gym.create_actor(env, tablet_asset, gymapi.Transform(p=gymapi.Vec3(0, 1.18, 1.0678), r=TABLET_ROT), "tablet", 0, 0)

# 튀김 준비/완료 거치대(fry_stand, STEP→URDF 변환). 오른쪽 작업대(worktable2_right) 위에 얹음.
# 충돌은 fry_stand.urdf에서 박스 4개(바닥판/앞쪽 물결벽/고리 2개)로 직접 근사해둠(원본 메쉬 VHACD가
# 얇은 바닥판+고리를 하나의 볼록껍질로 뭉쳐 바닥판 중앙이 실제보다 10cm 뜨는 문제가 있었음) → vhacd 불필요.
#
# 메쉬 분석 결과 뒷쪽(로컬 y:0.28~0.323)에 좌우 대칭 돌기(고리) 2개가 있고, 높이가 로컬 z=0~0.10로
# 정확히 작업대 턱의 높이차(메인 상판 z=0.85 → 뒷턱 z=0.95, 즉 10cm)와 일치한다. 즉 이 거치대는
# 작업대 메인 상판(z=0.85)에 올려놓고, 고리가 턱 위로 넘어가 턱의 발자국(월드 y:-1.173~-1.2)을
# 완전히 덮도록(고리 뒷면이 턱 뒤쪽 끝=바닥 끝 y=-1.2와 맞닿도록) 밀어 넣은 형태.
# worktable2_right는 180도 회전되어 있어 뒷턱이 월드 y:-1.173~-1.2에 있으므로, 거치대도 같은 180도 회전으로
# 고리 쪽(로컬 y=0.323)이 y=-1.2까지 밀려들어가게 배치.
stand2_opts = gymapi.AssetOptions(); stand2_opts.fix_base_link = True
fry_stand_asset = gym.load_asset(sim, asset_root, "urdf/fry_stand/fry_stand.urdf", stand2_opts)
FRY_STAND_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)
gym.create_actor(env, fry_stand_asset, gymapi.Transform(p=gymapi.Vec3(0, -0.877, 0.85), r=FRY_STAND_ROT), "fry_stand_right", 0, 0)

# fry_stand_right 바닥판 위에, 튀김기 위 바스켓들과 동일한 방식(눕혀서, 나란히, 손잡이 180도 회전)으로
# 배치. 바닥판 충돌이 이제 정확한 박스라 바닥판 윗면(월드 z=0.854)에 딱 맞게 놓인다.
# 앞쪽 물결벽(월드 y=-0.597)과 뒤쪽 고리(월드 y=-1.155~-1.2) 사이 안전지대(로컬 y:-0.19~0.19)에
# y방향으로 0.2m 간격을 두고 2개 배치 — 그 이상은 안전지대에 여유 있게 안 들어감.
#
# (근본 원인 추가 발견: fry_stand를 박스 충돌로 바꿔도 바스켓이 여전히 10cm 떠서 재조사한 결과,
#  진짜 원인은 worktable2.urdf의 충돌이 애초에 2단 형태 발견 전에 만든 "바운딩박스 전체(0.9x0.6x0.95)"
#  하나였던 것 — 메인 상판 영역까지도 전부 z=0.95로 충돌 처리되어 fry_stand가 그 안에 파묻혀 있었고,
#  fry_stand 위 물체는 fry_stand가 아니라 그 위의 작업대 충돌(0.95)에 얹혀 떴던 것. worktable2.urdf도
#  박스 2개(메인 상판 z:0~0.85 / 뒤쪽 턱 z:0~0.95)로 나눠 실제 2단 형태에 맞게 고쳤음.)
stand_flat_top_z = 0.85 + 0.004  # 박스 충돌 바닥판 윗면
NUM_STAND_BASKETS = 4
STAND_BASKET_SPACING = 0.2
# 기존 180도(손잡이 -X)에서 제자리 시계방향(위에서 볼 때) 90도 추가 회전 → 절대각 90도.
# 90도 회전하면 바스켓 긴 축(0.606m)이 Y방향이 되는데, 앞쪽 물결벽(y=-0.597)~뒤쪽 고리(y=-1.155)
# 안전지대 폭이 0.558m뿐이라 완전히는 못 들어가고 양쪽에 2.4cm씩 걸침(불가피, 물리적으로 꽉 낌).
# 안전지대 중앙(y=-0.876)에 두고 두 바스켓은 짧은 축(0.171m) 방향인 X로 나란히 배치했었는데,
# 로봇팔 쪽(+Y)으로 3cm 당김(뒤쪽 고리 겹침은 없어지고 앞쪽 물결벽 쪽만 살짝 걸침) → 오히려
# 드리프트/기울기 거의 0으로 더 안정적으로 안착함(시뮬레이션 확인).
STAND_BASKET_ROT2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi / 2)
stand_basket_center_y = -0.703
for i in range(NUM_STAND_BASKETS):
    local_x = (i - (NUM_STAND_BASKETS - 1) / 2) * STAND_BASKET_SPACING  # -0.1, +0.1
    pose = gymapi.Transform(
        p=gymapi.Vec3(local_x, stand_basket_center_y - BASKET_X_OFF, stand_flat_top_z - BASKET_Z_MIN),
        r=STAND_BASKET_ROT2,
    )
    gym.create_actor(env, basket_asset, pose, f"fry_stand_basket_{i}", 0, 0)

# 왼쪽 작업대(worktable2_left)에도 거치대 하나 더. worktable2_left는 무회전(뒷턱이 월드 y:1.173~1.2, +y쪽)이므로
# 거치대도 무회전으로 두면 고리(로컬 y=0.323)가 그대로 +y로 향해 턱과 맞음. 오른쪽과 동일하게 고리가
# 턱 발자국을 완전히 덮도록(고리 뒷면이 y=1.2에 맞닿도록) 배치.
gym.create_actor(env, fry_stand_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.877, 0.85)), "fry_stand_left", 0, 0)

# ============================================================ [3] 동역학 텐서(OSC)
gym.prepare_sim(sim)
arm.setup_osc()

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.6, 1.6, 1.6), gymapi.Vec3(0, 0, 0.9))

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

print("""
========== 두산 A0509 키보드 제어 ==========
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
