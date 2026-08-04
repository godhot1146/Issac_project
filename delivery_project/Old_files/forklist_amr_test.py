"""
forklift_full_test_suite.py
============================
ForkliftAMR + ForkliftMotionStep/ForkliftStepRunner 단독 테스트: 회전-이동 관련 전체 케이스 자동 검증.

케이스 1 (#1~8):  순수 직진 (facing과 axis가 일치) — FORWARD/BACKWARD × 4방향
케이스 2 (#9~16): 스텝 내부 축 전환 (axis_order, direction1/direction2 조합)
케이스 3 (#17~20): 스텝 간 전환 (이전 스텝 final_facing → 다음 스텝 initial_facing)
케이스 4 (#21~24): 큰 각도 회전 (180도 정반대, 90도 일반 회전)
"""

import os
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch

from forklift_amr_controller_v1 import ForkliftAMR, ForkliftMotionStep, ForkliftStepRunner

# =================================================================
# 1. 시뮬레이션 초기화
# =================================================================
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Forklift Full Rotation/Movement Test Suite")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
gym.set_light_parameters(sim, 0, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1.0, 0.0, -1.0))

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

env = gym.create_env(sim, gymapi.Vec3(-10, -10, 0), gymapi.Vec3(10, 10, 5), 1)

# =================================================================
# 2. 포크리프트만 스폰
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")

forklift_opts = gymapi.AssetOptions()
forklift_opts.fix_base_link = False
forklift_opts.replace_cylinder_with_capsule = True
forklift_asset = gym.load_asset(sim, asset_root, "urdf/forklift/forklift_v1.urdf", forklift_opts)

START_POSE = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.3))
forklift_handle = gym.create_actor(env, forklift_asset, START_POSE, "forklift_asset", -1, 1)

forklift_amr = ForkliftAMR(gym, sim, env, forklift_handle)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(4, 4, 4), gymapi.Vec3(0, 0, 0))

gym.prepare_sim(sim)

# =================================================================
# 3. 리셋 유틸리티
# =================================================================
root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)
forklift_actor_idx = gym.get_actor_index(env, forklift_handle, gymapi.DOMAIN_SIM)


def reset_forklift_to_origin():
    """포크리프트 위치/속도를 원점(yaw=0)으로 물리적 텔레포트, DOF/제어 상태도 초기화."""
    new_state = root_states[forklift_actor_idx].clone()
    new_state[0], new_state[1], new_state[2] = 0.0, 0.0, 0.15
    new_state[3], new_state[4], new_state[5], new_state[6] = 0.0, 0.0, 0.0, 1.0
    new_state[7:13] = 0.0
    root_states[forklift_actor_idx] = new_state

    idx_t = torch.tensor([forklift_actor_idx], dtype=torch.int32, device=root_states.device)
    gym.set_actor_root_state_tensor_indexed(
        sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(idx_t), 1
    )

    dof_count = forklift_amr.dof_count
    dof_states = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
    dof_states['pos'] = 0.0
    dof_states['vel'] = 0.0
    gym.set_actor_dof_states(env, forklift_handle, dof_states, gymapi.STATE_ALL)

    forklift_amr.current_v_lin = 0.0
    forklift_amr.current_v_ang = 0.0
    forklift_amr.target_yaw_fixed = None
    forklift_amr.locked_turn_direction = None
    forklift_amr.turn_complete = False
    forklift_amr.lift_height_ramped = 0.0
    forklift_amr.lift_height_target_desired = 0.0
    forklift_amr.dof_position_targets[:] = 0.0
    forklift_amr.dof_velocity_targets[:] = 0.0
    forklift_amr.apply_actuator_commands()


def get_pose_deg():
    curr_p, curr_r = forklift_amr.get_pose()
    yaw_deg = np.degrees(forklift_amr._get_current_yaw(curr_r))
    return curr_p, yaw_deg


def yaw_deg_diff(a, b):
    """두 각도(deg) 사이 최단 차이 (-180~180)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


# =================================================================
# 4. 공통 실행/판정 엔진
# =================================================================
TIMEOUT_FRAMES = 900          # 15초
ARRIVE_POS_TOLERANCE = 0.05   # m
ARRIVE_YAW_TOLERANCE = 2.0    # deg
JITTER_WARN_DEG = 1.0
DIVERGE_STREAK_LIMIT = 120    # 2초 연속 오차 증가 시 발산 판정

frame_count = 0
all_results = []


def _segment_axis_target(step, runner):
    """runner의 현재 sub_stage(SEGMENT1/2)에 대응하는 (axis_char, target_value)를 계산."""
    first_axis, second_axis = ('X', 'Y') if step.axis_order == 'x' else ('Y', 'X')
    first_target = step.target_x if first_axis == 'X' else step.target_y
    second_target = step.target_y if second_axis == 'Y' else step.target_x

    if runner.sub_stage == ForkliftStepRunner.SUB_SEGMENT1:
        return first_axis, first_target
    elif runner.sub_stage == ForkliftStepRunner.SUB_SEGMENT2:
        return second_axis, second_target
    return None, None


def run_case(case_name, steps, note=""):
    """
    steps(단일 또는 다중 ForkliftMotionStep 리스트)를 실행하고,
    발산/타임아웃/정지흔들림/최종위치오차/최종각도오차를 종합 판정.
    """
    global frame_count

    print(f"\n{'='*70}\n[TEST START] {case_name}  {note}\n{'='*70}")
    reset_forklift_to_origin()

    runner = ForkliftStepRunner(forklift_amr, steps, context=None)
    start_frame = frame_count

    yaw_samples_during_hold = []
    diverged = False
    diverge_reason = ""
    prev_axis_error = None
    divergence_streak = 0

    while not runner.done and not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)

        gym.refresh_actor_root_state_tensor(sim)

        current_step = runner._current_step()
        runner.update(frame_count)

        # 발산 감지 (직진 구간에서만)
        axis_char, target_val = _segment_axis_target(current_step, runner)
        if axis_char is not None:
            curr_p, _ = forklift_amr.get_pose()
            current_val = curr_p['x'] if axis_char == 'X' else curr_p['y']
            axis_error = abs(target_val - current_val)

            if prev_axis_error is not None and axis_error > prev_axis_error + 0.001:
                divergence_streak += 1
            else:
                divergence_streak = 0
            prev_axis_error = axis_error

            if divergence_streak > DIVERGE_STREAK_LIMIT:
                diverged = True
                diverge_reason = f"이동 구간 오차 계속 증가 (axis_error={axis_error:.3f}m)"
                print(f"  ⚠️ [발산 의심] {diverge_reason}, frame={frame_count}")
                break
        else:
            prev_axis_error = None
            divergence_streak = 0

        # 정지(hold) 구간 yaw 흔들림 기록
        if runner.sub_stage == ForkliftStepRunner.SUB_HOLD:
            _, yaw_deg = get_pose_deg()
            yaw_samples_during_hold.append(yaw_deg)

        if (frame_count - start_frame) > TIMEOUT_FRAMES:
            diverged = True
            diverge_reason = f"타임아웃 ({TIMEOUT_FRAMES}프레임 내 미완료, sub_stage={runner.sub_stage}, step_idx={runner.index})"
            print(f"  ⚠️ [타임아웃] {diverge_reason}")
            break

        gym.sync_frame_time(sim)
        frame_count += 1

    elapsed = frame_count - start_frame
    final_p, final_yaw = get_pose_deg()

    last_step = steps[-1]
    pos_error_x = abs(last_step.target_x - final_p['x'])
    pos_error_y = abs(last_step.target_y - final_p['y'])
    pos_error = max(pos_error_x, pos_error_y)

    expected_yaw = np.degrees(ForkliftAMR.direction_to_yaw(last_step.final_facing))
    yaw_error = abs(yaw_deg_diff(final_yaw, expected_yaw))

    yaw_jitter = (max(yaw_samples_during_hold) - min(yaw_samples_during_hold)) if yaw_samples_during_hold else 0.0

    passed = (
        not diverged
        and runner.done
        and pos_error < ARRIVE_POS_TOLERANCE
        and yaw_error < ARRIVE_YAW_TOLERANCE
        and yaw_jitter < JITTER_WARN_DEG
    )

    status = "✅ PASS" if passed else "❌ FAIL"
    fail_reasons = []
    if diverged:
        fail_reasons.append(diverge_reason)
    if not runner.done:
        fail_reasons.append("runner 미완료")
    if pos_error >= ARRIVE_POS_TOLERANCE:
        fail_reasons.append(f"위치오차={pos_error:.4f}m")
    if yaw_error >= ARRIVE_YAW_TOLERANCE:
        fail_reasons.append(f"각도오차={yaw_error:.2f}°")
    if yaw_jitter >= JITTER_WARN_DEG:
        fail_reasons.append(f"정지중 흔들림={yaw_jitter:.3f}°")

    print(f"  {status} | 소요 {elapsed}프레임({elapsed/60:.2f}s) | "
          f"위치오차={pos_error:.4f}m | 각도오차={yaw_error:.2f}° | "
          f"정지흔들림={yaw_jitter:.3f}° | 최종yaw={final_yaw:.2f}°")
    if fail_reasons:
        print(f"  실패 사유: {', '.join(fail_reasons)}")

    all_results.append({
        "case": case_name, "note": note, "passed": passed,
        "elapsed_frames": elapsed, "pos_error": pos_error,
        "yaw_error": yaw_error, "yaw_jitter": yaw_jitter,
        "fail_reasons": fail_reasons,
    })
    return passed


# =================================================================
# 5. 케이스 1 (#1~8): 순수 직진 — facing과 axis 일치
# =================================================================
MOVE_DISTANCE = 1.5

CASE1_LIST = [
    ("#1_+x_X_FWD", '+x', 'X', 'FORWARD'),
    ("#2_+x_X_BWD", '+x', 'X', 'BACKWARD'),
    ("#3_-x_X_FWD", '-x', 'X', 'FORWARD'),
    ("#4_-x_X_BWD", '-x', 'X', 'BACKWARD'),
    ("#5_+y_Y_FWD", '+y', 'Y', 'FORWARD'),
    ("#6_+y_Y_BWD", '+y', 'Y', 'BACKWARD'),
    ("#7_-y_Y_FWD", '-y', 'Y', 'FORWARD'),
    ("#8_-y_Y_BWD", '-y', 'Y', 'BACKWARD'),
]


def build_case1_step(name, facing, axis, direction):
    sign = +1.0 if direction == 'FORWARD' else -1.0
    facing_sign = +1.0 if facing.startswith('+') else -1.0
    delta = facing_sign * sign * MOVE_DISTANCE

    if axis == 'X':
        target_x, target_y = delta, 0.0
        axis_order = 'x'
    else:
        target_x, target_y = 0.0, delta
        axis_order = 'y'

    return ForkliftMotionStep(
        name, target_x=target_x, target_y=target_y,
        axis_order=axis_order,
        initial_facing=facing, direction1=direction,
        mid_facing=facing, direction2=direction,
        final_facing=facing,
        lift_height=None, hold_seconds=1.0,
    )


# =================================================================
# 6. 케이스 2 (#9~16): 스텝 내부 축 전환 — axis_order × direction1 × direction2
# =================================================================
CASE2_LIST = [
    ("#9_x_FWD_FWD",  'x', 'FORWARD',  'FORWARD'),
    ("#10_x_FWD_BWD", 'x', 'FORWARD',  'BACKWARD'),
    ("#11_x_BWD_FWD", 'x', 'BACKWARD', 'FORWARD'),
    ("#12_x_BWD_BWD", 'x', 'BACKWARD', 'BACKWARD'),
    ("#13_y_FWD_FWD",  'y', 'FORWARD',  'FORWARD'),
    ("#14_y_FWD_BWD", 'y', 'FORWARD',  'BACKWARD'),
    ("#15_y_BWD_FWD", 'y', 'BACKWARD', 'FORWARD'),
    ("#16_y_BWD_BWD", 'y', 'BACKWARD', 'BACKWARD'),
]


def build_case2_step(name, axis_order, direction1, direction2):
    """
    axis_order='x'면 1구간=X, 2구간=Y / axis_order='y'면 1구간=Y, 2구간=X.
    두 구간 모두 실제로 이동량이 발생하도록 target_x, target_y를 둘 다 원점과 다르게 설정.
    facing은 각 구간의 실제 이동 방향과 일치하도록 direction으로부터 역산.
    """
    d = MOVE_DISTANCE

    if axis_order == 'x':
        first_axis, second_axis = 'X', 'Y'
    else:
        first_axis, second_axis = 'Y', 'X'

    # 1구간: FORWARD면 +방향으로 진행 (facing=+축), BACKWARD면 facing=+축 유지한 채 -로 후진
    #        -> 항상 facing은 +축으로 통일하고 direction만 다르게 하여 "같은 facing, 다른 방향" 조합을 검증
    first_facing = '+' + first_axis.lower()
    second_facing = '+' + second_axis.lower()

    first_sign = +1.0 if direction1 == 'FORWARD' else -1.0
    second_sign = +1.0 if direction2 == 'FORWARD' else -1.0

    first_delta = first_sign * d
    second_delta = second_sign * d

    if first_axis == 'X':
        target_x = first_delta
        target_y = second_delta
    else:
        target_y = first_delta
        target_x = second_delta

    return ForkliftMotionStep(
        name, target_x=target_x, target_y=target_y,
        axis_order=axis_order,
        initial_facing=first_facing, direction1=direction1,
        mid_facing=second_facing, direction2=direction2,
        final_facing=second_facing,
        lift_height=None, hold_seconds=1.0,
    )


# =================================================================
# 7. 케이스 3 (#17~20): 스텝 간 전환 — 이전 스텝 final_facing -> 다음 스텝 initial_facing
# =================================================================
def build_case3_steps():
    """
    2-스텝 시퀀스 4개를 만든다. 각 시퀀스는:
      STEP_A: 원점에서 한쪽으로 이동, 특정 final_facing으로 종료
      STEP_B: STEP_A의 final_facing을 그대로(또는 다르게) initial_facing으로 받아 다음 이동
    반환: [(case_name, [step_a, step_b], note), ...]
    """
    cases = []

    # #17: 동일 facing(+x)으로 종료 -> 동일 facing으로 전진 이어가기
    step_a = ForkliftMotionStep(
        "17A", target_x=1.0, target_y=0.0, axis_order='x',
        initial_facing='+x', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x', hold_seconds=0.2,
    )
    step_b = ForkliftMotionStep(
        "17B", target_x=2.0, target_y=0.0, axis_order='x',
        initial_facing='+x', direction1='FORWARD',
        mid_facing='+x', direction2='FORWARD',
        final_facing='+x', hold_seconds=1.0,
    )
    cases.append(("#17_same_facing_FWD_continue", [step_a, step_b],
                  "동일 facing(+x) 유지, 즉시 통과 후 전진"))

    # #18: 동일 facing(-x)으로 종료 -> 동일 facing 유지한 채 후진 (지난 실전 버그 케이스)
    step_a = ForkliftMotionStep(
        "18A", target_x=-1.0, target_y=0.0, axis_order='x',
        initial_facing='-x', direction1='FORWARD',
        mid_facing='-x', direction2='FORWARD',
        final_facing='-x', hold_seconds=0.2,
    )
    step_b = ForkliftMotionStep(
        "18B", target_x=-2.0, target_y=0.0, axis_order='x',
        initial_facing='-x', direction1='BACKWARD',
        mid_facing='-x', direction2='BACKWARD',
        final_facing='-x', hold_seconds=1.0,
    )
    cases.append(("#18_same_facing_BWD_continue", [step_a, step_b],
                  "동일 facing(-x) 유지, 즉시 통과 후 후진 (실전 버그 재현 후보)"))

    # #19: Y축 정렬(+y)로 종료 -> 다음 스텝은 -x로 회전 필요 + 후진
    step_a = ForkliftMotionStep(
        "19A", target_x=0.0, target_y=1.0, axis_order='y',
        initial_facing='+y', direction1='FORWARD',
        mid_facing='+y', direction2='FORWARD',
        final_facing='+y', hold_seconds=0.2,
    )
    step_b = ForkliftMotionStep(
        "19B", target_x=-1.0, target_y=1.0, axis_order='x',
        initial_facing='-x', direction1='BACKWARD',
        mid_facing='-x', direction2='BACKWARD',
        final_facing='-x', hold_seconds=1.0,
    )
    cases.append(("#19_diff_facing_rotate_then_BWD", [step_a, step_b],
                  "+y 종료 -> -x로 회전 필요 + 후진"))

    # #20: -y 정렬로 종료 -> 다음 스텝은 +y로 180도 회전 필요 + 전진
    step_a = ForkliftMotionStep(
        "20A", target_x=0.0, target_y=-1.0, axis_order='y',
        initial_facing='-y', direction1='FORWARD',
        mid_facing='-y', direction2='FORWARD',
        final_facing='-y', hold_seconds=0.2,
    )
    step_b = ForkliftMotionStep(
        "20B", target_x=0.0, target_y=-2.0, axis_order='y',
        initial_facing='+y', direction1='FORWARD',
        mid_facing='+y', direction2='FORWARD',
        final_facing='+y', hold_seconds=1.0,
    )
    cases.append(("#20_diff_facing_180rotate_then_FWD", [step_a, step_b],
                  "-y 종료 -> +y로 180도 회전 필요 + 전진"))

    return cases


# =================================================================
# 8. 케이스 4 (#21~24): 큰 각도 회전 (이동 없이 순수 회전만)
# =================================================================
def build_case4_step(name, start_facing, target_facing):
    """
    이동 없이(target=원점 그대로) 회전만 검증.
    initial_facing=start_facing으로 먼저 정렬(SUB_INITIAL_FACING),
    이동 구간은 거리 0이라 즉시 통과, final_facing=target_facing에서
    실제 회전량(90도 또는 180도)을 검증.
    """
    return ForkliftMotionStep(
        name, target_x=0.0, target_y=0.0, axis_order='x',
        initial_facing=start_facing, direction1='FORWARD',
        mid_facing=start_facing, direction2='FORWARD',
        final_facing=target_facing,
        lift_height=None, hold_seconds=1.0,
    )


CASE4_LIST = [
    ("#21_+x_to_-x_180", '+x', '-x', "정확히 반대 방향 180도 회전"),
    ("#22_+y_to_-y_180", '+y', '-y', "정확히 반대 방향 180도 회전"),
    ("#23_+x_to_+y_90",  '+x', '+y', "일반적인 90도 회전"),
    ("#24_-x_to_-y_90",  '-x', '-y', "일반적인 90도 회전(다른 사분면)"),
]


# =================================================================
# 9. 전체 실행
# =================================================================
print("\n\n" + "#" * 70)
print("# 케이스 1: 순수 직진 (#1~8)")
print("#" * 70)
for name, facing, axis, direction in CASE1_LIST:
    step = build_case1_step(name, facing, axis, direction)
    run_case(name, [step], note=f"facing={facing} axis={axis} dir={direction}")

print("\n\n" + "#" * 70)
print("# 케이스 2: 스텝 내부 축 전환 (#9~16)")
print("#" * 70)
for name, axis_order, direction1, direction2 in CASE2_LIST:
    step = build_case2_step(name, axis_order, direction1, direction2)
    run_case(name, [step], note=f"axis_order={axis_order} dir1={direction1} dir2={direction2}")

print("\n\n" + "#" * 70)
print("# 케이스 3: 스텝 간 전환 (#17~20)")
print("#" * 70)
for name, steps, note in build_case3_steps():
    run_case(name, steps, note=note)

print("\n\n" + "#" * 70)
print("# 케이스 4: 큰 각도 회전 (#21~24)")
print("#" * 70)
for name, start_facing, target_facing, note in CASE4_LIST:
    step = build_case4_step(name, start_facing, target_facing)
    run_case(name, [step], note=note)

# =================================================================
# 10. 최종 요약
# =================================================================
print(f"\n\n{'='*70}\n[전체 결과 요약]\n{'='*70}")
pass_count = sum(1 for r in all_results if r["passed"])
for r in all_results:
    mark = "✅" if r["passed"] else "❌"
    reason_str = f" | 실패사유: {', '.join(r['fail_reasons'])}" if r["fail_reasons"] else ""
    print(f"  {mark} {r['case']:28s} {r['note']:45s} | "
          f"{r['elapsed_frames']:4d}f | pos_err={r['pos_error']:.4f} | "
          f"yaw_err={r['yaw_error']:.2f}° | jitter={r['yaw_jitter']:.3f}°{reason_str}")

print(f"\n총 {len(all_results)}개 중 {pass_count}개 통과, {len(all_results) - pass_count}개 실패\n")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)