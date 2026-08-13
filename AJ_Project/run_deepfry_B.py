"""
run.py — 두산 A0509 키보드 다중모드 제어 (JSC / TSC / OSC)

씬: 선반(고정) + A0509 팔(선반 상단 z=0.810에 고정 장착) + 에어 컴프레셔(선반 안).
제어: 실행 중 키로 모드를 바꿔가며 직접 조작.

  ┌─────────────── 키 맵 ───────────────┐
  │ [모드]  1: JSC(관절)  2: TSC(좌표+IK)  3: OSC(좌표+동역학)
  │ [좌표이동 · TSC/OSC]  W/S: X±   A/D: Y±   Q/E: Z±
  │ [관절이동 · JSC]      J/L: 관절 선택   U/O: 선택관절 각도±
  │ [공통]  R: 홈 자세    (뷰어 창 닫기: 종료)
  └──────────────────────────────────────┘

제어 로직은 controllers/doosan_controller.py (JSC/TSC/OSC 통합).
좌표 목표는 '로봇 베이스 기준'. OSC는 월드 텐서를 쓰므로 장착높이(+0.810) 보정해 넘긴다.

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

# 바닥 두께가 5t(0.005m)로 변경되어 기준 상단 높이 상승
FLOOR_T = 0.005 
Z_GROUND = FLOOR_T

BASE_Z     = 0.805 + Z_GROUND # 팔 장착 높이 (다이 0.805 + 바닥두께 0.005)
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

# [바닥면 및 레이아웃 빈틈 최소화 최적화]
# 풋프린트: 다이[-0.45~0.45, -0.3~0.3], 튀김기[0.95~1.55, -0.45~0.45], 좌/우 테이블[-0.45~0.45, ±0.3~±0.9]
# 전체 레이아웃 바운딩박스: X [-0.45 ~ 1.55] (폭 2.0m), Y [-0.9 ~ 0.9] (폭 1.8m)
FLOOR_W = 2.000   # 가로(X), m
FLOOR_D = 1.800   # 세로(Y), m
FLOOR_X_OFFSET = 0.55 # X 바운딩박스 중앙점 (-0.45 + 1.55) / 2

floor_opts = gymapi.AssetOptions(); floor_opts.fix_base_link = True
floor_asset = gym.create_box(sim, FLOOR_W, FLOOR_D, FLOOR_T, floor_opts)
floor_actor = gym.create_actor(
    env, floor_asset,
    gymapi.Transform(p=gymapi.Vec3(FLOOR_X_OFFSET, 0, FLOOR_T / 2)), # Z를 0.0025에 두어 윗면이 0.005(5t)가 되도록 함
    "floor", 0, 0,
)
gym.set_rigid_body_color(env, floor_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 1.0, 1.0))

# 로봇팔 거치대
STAND_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi / 2)
def _rot90_xy(x, y):
    return -y, x
stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True
stand_asset = gym.load_asset(sim, asset_root, "urdf/robot_cabinetnplate/robot_cabinetnplate.urdf", stand_opts)
gym.create_actor(env, stand_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, Z_GROUND), r=STAND_ROT), "stand", 0, 0)

# 작업대 (worktable2)
table_opts = gymapi.AssetOptions(); table_opts.fix_base_link = True
table_asset = gym.load_asset(sim, asset_root, "urdf/worktable2/worktable2.urdf", table_opts)

# 튀김기 (X축 스탠드 앞 edge[0.45]로부터 50cm(0.5m) 이격 -> X [0.95 ~ 1.55], 중심 1.25)
fryer_opts = gymapi.AssetOptions(); fryer_opts.fix_base_link = True
fryer_opts.vhacd_enabled = True
fryer_opts.vhacd_params = gymapi.VhacdParams()
fryer_opts.vhacd_params.resolution = 300000
fryer_asset = gym.load_asset(sim, asset_root, "urdf/Fryer/Fryer.urdf", fryer_opts)
FRYER_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
fryer_cx = 1.25
fryer_cy = 0.0
fryer_p = gymapi.Vec3(fryer_cx, fryer_cy, Z_GROUND)
gym.create_actor(env, fryer_asset, gymapi.Transform(p=fryer_p, r=FRYER_ROT), "Fryer", 0, 0)

# 프라이어 바스켓
basket_opts = gymapi.AssetOptions()
basket_opts.vhacd_enabled = True
basket_opts.vhacd_params = gymapi.VhacdParams()
basket_opts.vhacd_params.resolution = 300000
basket_asset = gym.load_asset(sim, asset_root, "urdf/fryer_basket/fryer_basket.urdf", basket_opts)

BASKET_X_OFF = -0.0475
BASKET_HALF_LEN = 0.2797
BASKET_Z_MIN = -0.1719
BASKET_W = 0.165
BASKET_HANDLE_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)

# 선반 대용 작업대 2개 (빈틈 없이 다이 측면에 밀착)
right_cx, right_cy = 0.0, -0.6
gym.create_actor(env, table_asset, gymapi.Transform(p=gymapi.Vec3(right_cx - 0.45, right_cy - 0.3, Z_GROUND)), "worktable2_right", 0, 0)
LEFT_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)
left_cx, left_cy = 0.0, 0.6
gym.create_actor(env, table_asset, gymapi.Transform(p=gymapi.Vec3(left_cx + 0.45, left_cy + 0.3, Z_GROUND), r=LEFT_ROT), "worktable2_left", 0, 0)

# 로봇 팔 (고정 베이스 0.810)
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf="urdf/doosan_a0509/a0509.urdf",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0.160, 0, BASE_Z)),
)

# 컴프레셔 (다이 수납장)
comp_opts = gymapi.AssetOptions(); comp_opts.fix_base_link = True
comp_asset = gym.load_asset(sim, asset_root, "urdf/air_compressor/air_compressor.urdf", comp_opts)
COMP_BOTTOM_Z = Z_GROUND + 0.02 - (-0.267)
comp_x, comp_y = _rot90_xy(-0.22925, 0.1009)
gym.create_actor(env, comp_asset, gymapi.Transform(p=gymapi.Vec3(comp_x, comp_y, COMP_BOTTOM_Z), r=STAND_ROT), "air_compressor", 0, 0)

# 미니 PC
pc_opts = gymapi.AssetOptions(); pc_opts.fix_base_link = True
pc_asset = gym.load_asset(sim, asset_root, "urdf/mini_pc/mini_pc.urdf", pc_opts)
pc_x, pc_y = _rot90_xy(0, 0.32)
gym.create_actor(env, pc_asset, gymapi.Transform(p=gymapi.Vec3(pc_x, pc_y, Z_GROUND + 0.12), r=STAND_ROT), "mini_pc", 0, 0)

# 로봇 컨트롤러
ctrl_opts = gymapi.AssetOptions(); ctrl_opts.fix_base_link = True
ctrl_asset = gym.load_asset(sim, asset_root, "urdf/doosan_controller/doosan_controller.urdf", ctrl_opts)
CTRL_ROT = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
CTRL_BOTTOM_Z = Z_GROUND - (-0.1991)
CTRL_X, CTRL_Y = 0.303 + 0.0875, 0.0 + 0.202
gym.create_actor(env, ctrl_asset, gymapi.Transform(p=gymapi.Vec3(CTRL_X, CTRL_Y, CTRL_BOTTOM_Z), r=CTRL_ROT), "doosan_controller", 0, 0)

# 태블릿
tablet_opts = gymapi.AssetOptions(); tablet_opts.fix_base_link = True
tablet_asset = gym.load_asset(sim, asset_root, "urdf/tablet/tablet.urdf", tablet_opts)
TABLET_ROT = gymapi.Quat(0.5, -0.5, 0.5, 0.5)
gym.create_actor(env, tablet_asset, gymapi.Transform(p=gymapi.Vec3(0, 1.18, 1.0678 + Z_GROUND)), "tablet", 0, 0)

# 튀김 준비/완료 거치대 (오른쪽)
stand2_opts = gymapi.AssetOptions(); stand2_opts.fix_base_link = True
fry_stand_asset = gym.load_asset(sim, asset_root, "urdf/fry_stand/fry_stand.urdf", stand2_opts)
fry_stand_right_y = 2 * right_cy - (-0.58)
gym.create_actor(env, fry_stand_asset, gymapi.Transform(p=gymapi.Vec3(0, fry_stand_right_y, 0.85 + Z_GROUND)), "fry_stand_right", 0, 0)

# 거치대 위 바스켓
stand_flat_top_z = 0.85 + Z_GROUND + 0.004
NUM_STAND_BASKETS = 4
STAND_BASKET_SPACING = 0.2
STAND_BASKET_ROT2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
stand_basket_center_y = -0.38
stand_basket_y = 2 * right_cy - (stand_basket_center_y - BASKET_X_OFF)

for i in range(NUM_STAND_BASKETS):
    local_x = (i - (NUM_STAND_BASKETS - 1) / 2) * STAND_BASKET_SPACING 
    pose = gymapi.Transform(
        p=gymapi.Vec3(local_x, stand_basket_y, stand_flat_top_z - BASKET_Z_MIN),
        r=STAND_BASKET_ROT2,
    )
    gym.create_actor(env, basket_asset, pose, f"fry_stand_basket_{i}", 0, 0)

# 튀김 거치대 (왼쪽)
fry_stand_left_y = 2 * left_cy - 0.58
gym.create_actor(env, fry_stand_asset, gymapi.Transform(p=gymapi.Vec3(0, fry_stand_left_y, 0.85 + Z_GROUND), r=LEFT_ROT), "fry_stand_left", 0, 0)

# ============================================================ [3] 동역학 텐서(OSC)
gym.prepare_sim(sim)
arm.setup_osc()

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.6, 1.6, 1.6), gymapi.Vec3(0, 0, 0.9))

# 키보드 텔레오프 (JSC/TSC/OSC, 꾹 누르면 연속, TSC 자세 유지) — controllers/arm_keyboard_teleop.py
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