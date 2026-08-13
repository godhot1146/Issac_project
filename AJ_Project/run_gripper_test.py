"""
run_gripper_test.py — A0509 + two_finger_gripper 부착 테스트.

두산 A0509 팔 손목(link_6)에 two_finger_gripper를 용접한 결합 URDF
(urdf/a0509_two_finger_gripper/a0509_two_finger_gripper.urdf)를 스폰하고,
키보드로 팔 이동(TSC) + 그리퍼 열기/닫기를 테스트한다.

*** 미검증 스크립트 — 이 환경엔 GPU/디스플레이/issac_env가 없어 실행해서 확인하지
못했다. 그리퍼가 손목에 파묻히거나 엉뚱한 방향을 보면 a0509_two_finger_gripper.urdf의
link_6-gripper_base_link 조인트 origin/rpy부터 조정할 것 (파일 상단 주석 참고). ***

키:
  W/S: X±   A/D: Y±   Q/E: Z±   (TSC, 손끝 좌표 이동, 자세 유지 없음)
  N  : 그리퍼 닫기(꾹 누르면 연속)     M: 그리퍼 열기(꾹 누르면 연속)
  R  : 홈 자세

실행:  conda activate issac_env  &&  python run_gripper_test.py
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil   # torch보다 먼저

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers"))
from doosan_controller import DoosanController

from asset_config import get_asset_root
asset_root = get_asset_root()

BASE_Z = 0.3
CART_STEP = 0.004
HOME_Q = np.zeros(6, dtype=np.float32)
GRIPPER_STEP = 0.0003                # 프레임당 이동량(꾹 누르면 연속)
# two_finger_gripper.urdf의 joint limit과 동일: MIN=완전 열림(슬라이드 슬롯 바깥쪽 벽에 닿음,
# 간격 22mm), MAX=완전 닫힘(간격 0mm)
GRIPPER_MIN, GRIPPER_MAX = -0.010, 0.001

# ============================================================ [1] 시뮬 + 씬
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="A0509 + two_finger_gripper test")
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

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# 팔+그리퍼 결합 URDF를 하나의 액터로 스폰. IK는 그리퍼 DOF가 안 섞이도록 순수 팔
# URDF(ik_urdf) 기준으로 별도 계산한다 — DoosanController가 이미 지원하는 파라미터.
arm = DoosanController(
    gym, sim, env, asset_root,
    urdf="urdf/a0509_two_finger_gripper/a0509_two_finger_gripper.urdf",
    ik_urdf="urdf/doosan_a0509/a0509.urdf",
    ee_link="link_6",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0, 0, BASE_Z)),
)

dof_dict = gym.get_asset_dof_dict(arm.asset)
LEFT_IDX = dof_dict["gripper_left_finger_joint"]
RIGHT_IDX = dof_dict["gripper_right_finger_joint"]
print("DOF dict:", dof_dict)

# ============================================================ [2] 뷰어 + 키
gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.0, -1.0, 0.8), gymapi.Vec3(0, 0, 0.4))

keymap = {
    gymapi.KEY_W: "x+", gymapi.KEY_S: "x-",
    gymapi.KEY_A: "y+", gymapi.KEY_D: "y-",
    gymapi.KEY_Q: "z+", gymapi.KEY_E: "z-",
    gymapi.KEY_N: "grip_close", gymapi.KEY_M: "grip_open",
    gymapi.KEY_R: "home",
}
for k, a in keymap.items():
    gym.subscribe_viewer_keyboard_event(viewer, k, a)

print("""
===== A0509 + two_finger_gripper 테스트 =====
 W/S:X±  A/D:Y±  Q/E:Z±   (TSC, 손끝 좌표 이동)
 N: 그리퍼 닫기(꾹)   M: 그리퍼 열기(꾹)
 R: 홈 자세
==============================================""")

# 홈 자세로 시작 + TSC 목표를 현재 손끝에 동기화(튐 방지)
arm.go_joints(HOME_Q)
for _ in range(30):
    gym.simulate(sim); gym.fetch_results(sim, True)
cart_target = np.array(arm.current_tcp(), dtype=np.float32)
gripper_pos = GRIPPER_MIN

# 진단용: 스폰+30스텝 정착 후 그리퍼 각 링크의 실제 월드 좌표를 출력.
# link_6 근처(대략 BASE_Z=0.3 + 팔 길이 이내)에 있어야 정상. 값이 비정상적으로 크거나
# NaN이면 손목-그리퍼 용접부 콜리전이 겹쳐서 스폰 직후 튕겨나간(explosion) 것.
rb_dict = gym.get_actor_rigid_body_dict(env, arm.actor)
rb_states = gym.get_actor_rigid_body_states(env, arm.actor, gymapi.STATE_POS)
for name in ("link_6", "gripper_base_link", "gripper_left_finger_link", "gripper_right_finger_link"):
    idx = rb_dict[name]
    pos = rb_states["pose"]["p"][idx]
    print(f"[진단] {name} world pos = ({pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f})")

held = set()
_CONTINUOUS = {"x+", "x-", "y+", "y-", "z+", "z-", "grip_close", "grip_open"}

step = 0
while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        a = e.action
        if a in _CONTINUOUS:
            (held.add if e.value > 0 else held.discard)(a)
            continue
        if e.value <= 0:
            continue
        if a == "home":
            arm.go_joints(HOME_Q)
            for _ in range(30):
                gym.simulate(sim); gym.fetch_results(sim, True)
            cart_target = np.array(arm.current_tcp(), dtype=np.float32)
            print("[홈]")

    if held & {"x+", "x-", "y+", "y-", "z+", "z-"}:
        for a in held:
            if a in ("x+", "x-", "y+", "y-", "z+", "z-"):
                axis = {"x": 0, "y": 1, "z": 2}[a[0]]
                cart_target[axis] += CART_STEP if a[1] == "+" else -CART_STEP
        arm.go_cartesian(cart_target)

    if "grip_close" in held or "grip_open" in held:
        if "grip_close" in held:
            gripper_pos = min(GRIPPER_MAX, gripper_pos + GRIPPER_STEP)
        if "grip_open" in held:
            gripper_pos = max(GRIPPER_MIN, gripper_pos - GRIPPER_STEP)
        arm.set_extra_dof(LEFT_IDX, gripper_pos)
        arm.set_extra_dof(RIGHT_IDX, gripper_pos)

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    if step % 60 == 0:
        print(f"[TSC] 목표={np.round(cart_target,3)} 손끝={np.round(arm.current_tcp(),3)}  grip={gripper_pos:.4f}")
    step += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
