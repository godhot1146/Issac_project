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

from asset_config import get_asset_root
asset_root = get_asset_root()   # 컴퓨터마다 에셋 위치 자동 탐색/저장 (asset_config.py 참고)

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
# ┌── 스폰(배치) 전체 그림 ─────────────────────────────────────────────┐
# │ acquire_gym()  : gym 핸들 획득(모든 API의 진입점)                    │
# │ create_sim()   : 물리 세계 1개 생성(중력·타임스텝·PhysX 설정 포함)   │
# │ add_ground()   : 그 세계에 무한 바닥면(z=0) 부착 → 모든 env가 공유    │
# │ create_env()   : 액터를 담을 '방' 생성 (이 파일은 1개 → env원점=월드)│
# │ load_asset()   : URDF를 파싱해 sim에 '에셋 템플릿' 등록 (형상·관성)  │
# │ create_actor() : 그 템플릿을 env 안에 '실체(액터)'로 찍음 + 위치 지정 │
# └─────────────────────────────────────────────────────────────────────┘
# 즉 load_asset=붕어빵 '틀' 한 번 만들기 / create_actor=그 틀로 '붕어빵' 찍기.
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
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)   # 물리세계 1개
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)   # 바닥면 법선 = +Z (Z-up)
gym.add_ground(sim, pp)                                        # z=0 무한 바닥 부착(모든 env 공유)

# ============================================================ [2] 씬
# create_env(sim, lower, upper, num_per_row): 액터를 담을 방 1개.
#   lower/upper = 이 env의 경계상자(여러 env일 때 격자 간격·시각화용). num=1이라 env원점=월드(0,0,0).
env = gym.create_env(sim, gymapi.Vec3(-2.5, -2.5, 0), gymapi.Vec3(2.5, 2.5, 2), 1)

# ── 스폰 3단계(설비마다 반복) ────────────────────────────────────────────
#   ① AssetOptions: 로드 옵션. fix_base_link=True면 base_link를 월드에 '용접'(중력·충돌로 안 움직임).
#   ② load_asset(sim, asset_root, "urdf경로", opts): URDF 파싱 → 에셋 '템플릿' 반환(형상·관성 등록).
#   ③ create_actor(env, 템플릿, Transform, 이름, group, filter): 템플릿을 env에 실체로 찍음.
#        · Transform.p = 위치 Vec3(x,y,z)  [env로컬 ≈ 월드],  Transform.r = 회전 Quat(생략시 무회전)
#        · group  = 충돌 그룹(같은 값끼리만 충돌 계산),  filter = 비트마스크(끼리 충돌 제외)
# 스탠드+팔+컴프레셔는 그릴/완료존/준비존 3곳에서 등거리인 RIG_X,RIG_Y로 이동 배치.
stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True   # ① 바닥 고정 설비
stand_asset = gym.load_asset(sim, asset_root, "urdf/a0509_stand/a0509_stand.urdf", stand_opts)  # ② 템플릿
gym.create_actor(env, stand_asset, gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, 0)), "stand", 0, 0)  # ③ 배치

# 팔은 DoosanController가 위 ①②③(load+create_actor)을 내부에서 대신 해줌 → 여기선 파라미터만 넘김.
# 스탠드 상판(BASE_Z=0.81) 높이에 고정 장착. (OSC 동역학이 깔끔하도록 고정베이스 순수팔)
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf="urdf/doosan_a0509/a0509.urdf",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, BASE_Z)),   # 팔 base_link를 상판 높이에
)

# 컴프레셔: 스탠드와 같은 XY에, 살짝 띄워(z=0.02) 배치. (①②③ 동일 패턴)
comp_opts = gymapi.AssetOptions(); comp_opts.fix_base_link = True
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(RIG_X, RIG_Y, 0.02)), "air_compressor", 0, 0)

# ---- 구이 도면(AJ_4종(구이)_조리솔루션) 기준 배치 (그릴/완료/준비존은 방 경계 고정, 로봇과 무관) ----
# 그릴러: 설치공간 경계의 왼쪽-상단(북서) 모서리에 딱 맞게 — 뒷면을 벽(ROOM_WALL_Y)에,
# 왼쪽면을 경계 왼쪽(ROOM_X_MIN)에 붙임. 손잡이(로컬 y=-0.095~0)는 반경 안쪽으로 살짝 걸침.
grill_opts = gymapi.AssetOptions(); grill_opts.fix_base_link = True   # 그릴=고정 설비
grill_asset = gym.load_asset(sim, asset_root, "urdf/grill/grill.urdf", grill_opts)
gym.create_actor(env, grill_asset,
                  gymapi.Transform(p=gymapi.Vec3(ROOM_X_MIN, ROOM_WALL_Y - 0.6, 0.0)), "grill", 0, 0)

# 여기선 템플릿(asset)만 미리 로드해두고, 실제 배치(create_actor)는 아래 spawn_zone()이 존마다 반복.
#   → 같은 템플릿 하나로 완료존/준비존에 여러 번 찍는 게 load_asset/create_actor 분리의 이점.
# 고정 여부가 핵심: 테이블·랙은 고정(True), 바스켓만 이동체(False=중력/그리퍼에 반응).
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
BASKET_LEN = 0.532   # grill_basket 길이(DoubleGrillBasket.step 개정판, 이전 0.537에서 변경)
# grill_rack 테두리의 사다리꼴 홈(바스켓 손잡이가 걸리는 자리) 2개 간격 — 삼각형 위치 분석으로
# 실측(로컬 x=±0.225, 회전 후 dy축에 대응) → 기존 ±0.2 대신 이 값을 써야 홈에 정확히 걸림.
NOTCH_SPACING = 0.225

def spawn_zone(center, name_prefix, basket_flip, table_rot=ROT270):
    """work_table(받침대) → grill_rack(거치대) → 바스켓 2개, 3단으로 쌓아 배치.
    center = (Cx,Cy): 이 존의 X,Y 중심. table_rot(ROT270/ROT90)에 따라 테이블/거치대가
    같은 자리(footprint)에서 180도 반대 방향을 보도록 spawn 오프셋만 바뀐다
    (ROT270: table(0.3,-0.45)/rack(0.0215,0), ROT90: table(-0.3,0.45)/rack(-0.0215,0))."""
    cx, cy = center
    # 회전(r=table_rot)을 주면 메쉬 원점이 회전축이 되어 위치가 틀어지므로, 그만큼 spawn p를
    # 오프셋(table_off/rack_off)해서 '같은 footprint 안'에 놓이게 보정한다(회전+평행이동 조합).
    if table_rot is ROT270:
        table_off, rack_off = (0.3, -0.45), (0.0215, 0.0)
    else:  # ROT90 — 방향 반대
        table_off, rack_off = (-0.3, 0.45), (-0.0215, 0.0)
    # 받침대: 바닥(z=0)에. Transform에 p(위치)와 r(회전)을 함께 지정.
    gym.create_actor(env, table_asset,
                      gymapi.Transform(p=gymapi.Vec3(cx - table_off[0], cy - table_off[1], 0.0), r=table_rot),
                      f"{name_prefix}_table", 0, 0)
    # 거치대: 받침대 상판(TABLE_TOP_Z) 위에 얹음 → z만 올려주면 3단 적층이 됨.
    gym.create_actor(env, rack_asset,
                      gymapi.Transform(p=gymapi.Vec3(cx - rack_off[0], cy - rack_off[1], TABLE_TOP_Z), r=table_rot),
                      f"{name_prefix}_rack", 0, 0)
    bx = (cx + BASKET_LEN / 2 if basket_flip else cx - BASKET_LEN / 2) + (-0.2 if basket_flip else 0.2)
    basket_rot = ROT180 if basket_flip else gymapi.Quat()  # gymapi.Quat() = identity(무회전)
    # 바스켓 2개: 랙 상판(RACK_TOP_Z) 위 3cm 띄워 떨어뜨림(이동체라 물리로 랙 홈에 안착).
    # 같은 basket_asset 템플릿을 dy만 바꿔 2번 create_actor → 이름은 유일해야 하므로 인덱스 붙임.
    for i, dy in enumerate([-NOTCH_SPACING, NOTCH_SPACING]):
        gym.create_actor(env, basket_asset, gymapi.Transform(
            p=gymapi.Vec3(bx, cy + dy, RACK_TOP_Z + 0.03), r=basket_rot),
            f"{name_prefix}_basket_{i}", 0, 0)

# 바스켓 손잡이(힌지+레버, 삼각형 면적분석으로 로컬 x=0.42~0.51 구간에 있음 확인)가 랙 테두리
# 홈 쪽으로 오도록 flip 지정 — 완료(서쪽 노치)는 flip=True, 준비(동쪽 노치)는 flip=False일 때
# 손잡이가 노치 가장자리에서 ~7~8cm 이내로 들어옴(반대 flip은 40cm+ 벌어짐).
spawn_zone(DONE_CENTER, "done", basket_flip=False, table_rot=ROT90)
spawn_zone(READY_CENTER, "ready", basket_flip=True)

# ============================================================ [3] 동역학 텐서(OSC)
# 중요: 모든 create_actor(스폰)가 끝난 뒤에 prepare_sim 호출 → 이후엔 액터 추가 불가.
#       prepare_sim이 물리 상태 텐서를 확정하고, 그걸 setup_osc가 받아 OSC 제어에 사용.
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

# 키보드 텔레오프 (JSC/TSC/OSC, 꾹 누르면 연속, TSC 자세 유지) — controllers/keyboard_teleop.py
from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop
teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)

step = 0
while not gym.query_viewer_has_closed(viewer):
    teleop.handle_and_apply()   # 이벤트 처리 + 연속동작 + 모드별 제어

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env, 4, ROOM_LINES, ROOM_COLORS)
    gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    step += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
