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

BASE_Z     = 0.81     # 팔 장착 높이(A0509_Stand 상판면)
CART_STEP  = 0.01     # 좌표 목표 이동 스텝(m)
JOINT_STEP = 0.05     # 관절 이동 스텝(rad)
HOME_Q     = np.array([0.0, 0.0, 1.2, 0.0, 1.0, 0.0], dtype=np.float32)

# ---- 구이 도면(AJ_4종(구이)_조리솔루션) 기준 설치공간/설비 좌표 ----
# 좌표계(로봇 베이스=원점 기준, 회전 전): +y=북(벽), -y=남(사람), +x=동(준비존), -x=서.
# 도면의 "A0509(900mm)" 원은 실측상 지름 900mm로 해석(반지름 0.45m).
R_REACH = 0.45   # 로봇 작업반경(도면 원 반지름)

ROOM_W = 1.85                       # 도면 폭(x)
ROOM_D = 2.255                      # 도면 깊이(y)
ROOM_WALL_Y  = R_REACH + 0.6        # 그릴러 뒷면(=벽)까지 거리 = 반경 + 그릴러 깊이
ROOM_FRONT_Y = ROOM_WALL_Y - ROOM_D # 사람쪽 개방 경계
ROOM_X_MIN, ROOM_X_MAX = -ROOM_W / 2, ROOM_W / 2

ROT90  = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), float(np.pi / 2))    # 북→서
ROT180 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), float(np.pi))       # 북→남
ROT270 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), float(-np.pi / 2))  # 북→동

# 설비 중심(고정, 방 경계 기준 — 로봇 위치와 무관):
GRILL_CENTER_X = ROOM_X_MIN + 0.75   # 그릴러(1.5m 폭) X중심

# 완료/준비 존: work_table(0.9x0.6x0.95m, 받침대) 위에 grill_rack(구이 준비,완료 거치대.step,
# 0.905x0.603x0.103m, 낮은 랙)을 얹고, 그 위에 바스켓을 놓는 3단 구조.
# work_table 로컬 원점=바닥 모서리(0~0.9,0~0.6). ROT270 적용 시 x:0~0.6,y:-0.9~0 (중심 (0.3,-0.45)).
# grill_rack 로컬 원점=x중심정렬(-0.4525~0.4525)/y비대칭(-0.28~0.323). ROT270 적용 시
# x:-0.28~0.323,y:-0.4525~0.4525 (중심 (0.0215,0)).
RACK_Y_MAX = 0.325   # 그릴러 손잡이(0.355)에서 3cm 여유 — 존 중심의 상단 기준
DONE_CENTER  = (ROOM_X_MIN + 0.3, RACK_Y_MAX - 0.45)   # 완료존 중심(work_table 기준): 서쪽 변
READY_CENTER = (ROOM_X_MAX - 0.3, RACK_Y_MAX - 0.45)   # 준비존 중심(work_table 기준): 동쪽 변

# 로봇(스탠드+팔+컴프레셔) 그룹 위치: A0509_Stand(0.6x0.9m)가 완료(-0.925~-0.325)/준비
# (0.325~0.925) 작업대 사이 0.65m 틈에 안 겹치게 들어가는 자리는 사실상 중앙(X=0)뿐이라
# (양쪽 여유 2.5cm) 오른쪽(그릴러 X중심 -0.175 → 0)으로 이동. Y는 기존과 동일.
RIG_X, RIG_Y = 0.0, READY_CENTER[1]

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
env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0), gymapi.Vec3(2.5, 2.5, 2), 1)

# 스탠드+팔+컴프레셔는 그릴/완료존/준비존 3곳에서 등거리인 RIG_X,RIG_Y로 이동 배치.
stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True
stand_asset = gym.load_asset(sim, asset_root, "urdf/a0509_stand/a0509_stand.urdf", stand_opts)
gym.create_actor(env, stand_asset, gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, 0)), "stand", 0, 0)

# 순수 A0509를 선반 상단(z=0.8)에 고정 장착 (OSC 동역학이 깔끔하도록 고정베이스 순수팔)
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf="urdf/doosan_a0509/a0509.urdf",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, BASE_Z)),
)

comp_opts = gymapi.AssetOptions(); comp_opts.fix_base_link = True
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, 0.02)), "air_compressor", 0, 0)

# ---- 구이 도면(AJ_4종(구이)_조리솔루션) 기준 배치 (그릴/완료/준비존은 방 경계 고정, 로봇과 무관) ----
# 그릴러: 설치공간 경계의 왼쪽-상단(북서) 모서리에 딱 맞게 — 뒷면을 벽(ROOM_WALL_Y)에,
# 왼쪽면을 경계 왼쪽(ROOM_X_MIN)에 붙임. 손잡이(로컬 y=-0.095~0)는 반경 안쪽으로 살짝 걸침.
grill_opts = gymapi.AssetOptions(); grill_opts.fix_base_link = True
grill_asset = gym.load_asset(sim, asset_root, "urdf/grill/grill.urdf", grill_opts)
gym.create_actor(env, grill_asset,
                  gymapi.Transform(p=gymapi.Vec3(ROOM_X_MIN, ROOM_WALL_Y - 0.6, 0.0)), "grill", 0, 0)

table_opts = gymapi.AssetOptions(); table_opts.fix_base_link = True
rack_opts = gymapi.AssetOptions(); rack_opts.fix_base_link = True
basket_opts = gymapi.AssetOptions(); basket_opts.fix_base_link = False  # 이동체(집었다 놓는 대상)
table_asset = gym.load_asset(sim, asset_root, "urdf/work_table/work_table.urdf", table_opts)
rack_asset = gym.load_asset(sim, asset_root, "urdf/grill_rack/grill_rack.urdf", rack_opts)
basket_asset = gym.load_asset(sim, asset_root, "urdf/grill_basket/grill_basket.urdf", basket_opts)

# work_table 실제 상판은 면적분석 결과 z=0.85 (바운딩박스 950mm는 얇은 테두리 립일 뿐,
# 면적 360cm^2로 하중면 아님) — 콜리전도 0.85로 낮춰서 반영. grill_rack도 마찬가지로
# 시각적으로는 103mm 바운딩박스지만 실바닥판은 0~2mm(핀 2개만 103mm까지 돌출, 면적<1cm^2라
# 하중면 아님) — 콜리전을 얇은 판(0~5mm)으로 수정했으므로 바스켓은 그 위(5mm)에 안착.
TABLE_TOP_Z = 0.85           # work_table 실제 상판 높이
RACK_TOP_Z = TABLE_TOP_Z + 0.005   # + grill_rack 바닥판 두께(0.005)

def spawn_zone(center, name_prefix, basket_flip, table_rot=ROT270):
    """work_table(받침대) → grill_rack(거치대) → 바스켓 2개, 3단으로 쌓아 배치.
    center = (Cx,Cy): 이 존의 X,Y 중심. table_rot(ROT270/ROT90)에 따라 테이블/거치대가
    같은 자리(footprint)에서 180도 반대 방향을 보도록 spawn 오프셋만 바뀐다
    (ROT270: table(0.3,-0.45)/rack(0.0215,0), ROT90: table(-0.3,0.45)/rack(-0.0215,0))."""
    cx, cy = center
    if table_rot is ROT270:
        table_off, rack_off = (0.3, -0.45), (0.0215, 0.0)
    else:  # ROT90 — 방향 반대
        table_off, rack_off = (-0.3, 0.45), (-0.0215, 0.0)
    gym.create_actor(env, table_asset,
                      gymapi.Transform(p=gymapi.Vec3(cx - table_off[0], cy - table_off[1], 0.0), r=table_rot),
                      f"{name_prefix}_table", 0, 0)
    gym.create_actor(env, rack_asset,
                      gymapi.Transform(p=gymapi.Vec3(cx - rack_off[0], cy - rack_off[1], TABLE_TOP_Z), r=table_rot),
                      f"{name_prefix}_rack", 0, 0)
    bx = cx + 0.537 / 2 if basket_flip else cx - 0.537 / 2
    basket_rot = ROT180 if basket_flip else gymapi.Quat()  # gymapi.Quat() = identity(무회전)
    for i, dy in enumerate([-0.2, 0.2]):
        gym.create_actor(env, basket_asset, gymapi.Transform(
            p=gymapi.Vec3(bx, cy + dy, RACK_TOP_Z + 0.03), r=basket_rot),
            f"{name_prefix}_basket_{i}", 0, 0)

spawn_zone(DONE_CENTER, "done", basket_flip=False, table_rot=ROT90)   # 방향 반대(요청 반영)
spawn_zone(READY_CENTER, "ready", basket_flip=True)

# ============================================================ [3] 동역학 텐서(OSC)
gym.prepare_sim(sim)
arm.setup_osc()

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3.2, -3.2, 2.6), gymapi.Vec3(0, -0.1, 0.5))

# 설치공간 경계(1850x2255mm) 표시용 사각형 라인 (매 프레임 그려줌)
ROOM_LINES = np.array([
    ROOM_X_MIN, ROOM_WALL_Y,  0.01,  ROOM_X_MAX, ROOM_WALL_Y,  0.01,
    ROOM_X_MAX, ROOM_WALL_Y,  0.01,  ROOM_X_MAX, ROOM_FRONT_Y, 0.01,
    ROOM_X_MAX, ROOM_FRONT_Y, 0.01,  ROOM_X_MIN, ROOM_FRONT_Y, 0.01,
    ROOM_X_MIN, ROOM_FRONT_Y, 0.01,  ROOM_X_MIN, ROOM_WALL_Y,  0.01,
], dtype=np.float32)
ROOM_COLORS = np.array([[1.0, 1.0, 0.0]] * 4, dtype=np.float32)

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
    gym.step_graphics(sim)
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env, 4, ROOM_LINES, ROOM_COLORS)
    gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    if step % 60 == 0:
        tcp = arm.current_tcp()
        extra = f"  [선택 joint_{sel_joint+1}]" if mode == "jsc" else ""
        print(f"[{mode.upper()}] 목표(베이스)={np.round(cart_target,3)} 손끝={np.round(tcp,3)}{extra}")
    step += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
