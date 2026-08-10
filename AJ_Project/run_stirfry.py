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

stand_opts = gymapi.AssetOptions(); stand_opts.fix_base_link = True
stand_asset = gym.load_asset(sim, asset_root, "urdf/robot_stand/robot_stand.urdf", stand_opts)
gym.create_actor(env, stand_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0)), "stand", 0, 0)

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

# ------------------------------------------------------------
# Bonitkit
# ------------------------------------------------------------

bonit_opts = gymapi.AssetOptions()
bonit_opts.fix_base_link = True

bonit_asset = gym.load_asset(
    sim,
    asset_root,
    "urdf/bonitkit/bonitkit.urdf",   # 실제 URDF 이름에 맞게 수정
    bonit_opts
)

bonit_pose = gymapi.Transform()
bonit_pose.p = gymapi.Vec3(0.8, 0.0, 0.0)

gym.create_actor(
    env,
    bonit_asset,
    bonit_pose,
    "bonitkit",
    0,
    0
)

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
