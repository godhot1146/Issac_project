"""
tests/test_lowamr.py — LowAMR 시나리오 전용 테스트

수동 텔레옵은 이미 검증된 Dual-AMR Teleop 스크립트를 그대로 쓰기로 했으므로, 이 스크립트는
main에서 실제로 쓰는 방식인 LowAmrMotionStep + LowAmrStepRunner 시나리오(리프트업+회전+Yaw-Lock
포함)가 독립 실행 환경에서도 끝까지 잘 도는지만 검증한다.

실행하면 바로 시나리오가 1회 자동 시작된다.
    ENTER : 시나리오 재시작 (전진+리프트업+회전+YawLock -> 복귀+리프트다운)
"""

from isaacgym import gymapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_common import create_minimal_sim, create_viewer, load_asset, main_loop
from low_amr_controller_v2 import LowAMR, LowAmrMotionStep, LowAmrStepRunner
from indy7_controller_v1 import StepContext


def build_scene():
    gym, sim, env, args = create_minimal_sim("LowAMR Scenario Test")
    viewer = create_viewer(gym, sim, env, cam_pos=(3.0, -3.0, 2.5), cam_target=(0.0, 0.0, 0.3))

    low_asset = load_asset(
        gym, sim, "urdf/low/v2/low_amr_v2.urdf",
        fix_base_link=False, replace_cylinder_with_capsule=True,
    )
    low_handle = gym.create_actor(
        env, low_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.3)), "low_amr_test", -1, 1
    )
    low_amr = LowAMR(gym, sim, env, low_handle)

    gym.prepare_sim(sim)

    return gym, sim, env, viewer, low_amr


def bind_keys(gym, viewer):
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, "start_scenario")


def build_scenario_runner(low_amr):
    """
    main의 low_amr_steps 패턴을 단독 실행용으로 축약: 전진+리프트업+회전판+Yaw-Lock 활성화
    -> 원위치 복귀+리프트다운+Yaw-Lock 해제. (main과 달리 wait_for/on_complete로 다른
    컨트롤러를 깨우지 않는다 — 단독 테스트라 필요 없음)
    """
    steps = [
        LowAmrMotionStep(
            "MOVE_AND_LIFT",
            target_x=1.2, target_y=0.0, axis_order='x',
            initial_facing='+x', direction1='FORWARD',
            mid_facing='+x', direction2='FORWARD',
            final_facing='+x',
            lift_height=0.16, lift_rotate_angle=0.5,
            lift_yaw_lock=True,
            hold_seconds=1.0,
        ),
        LowAmrMotionStep(
            "RETURN_AND_LOWER",
            target_x=0.0, target_y=0.0, axis_order='x',
            initial_facing='+x', direction1='BACKWARD',
            mid_facing='+x', direction2='BACKWARD',
            final_facing='+x',
            lift_height=0.0, reset_lift_plate=True,
            lift_yaw_lock=False,
            hold_seconds=1.0,
        ),
    ]
    return LowAmrStepRunner(low_amr, steps, context=StepContext(low_amr=low_amr))


def main():
    gym, sim, env, viewer, low_amr = build_scene()
    bind_keys(gym, viewer)

    scenario_runner = [build_scenario_runner(low_amr)]  # 실행하자마자 1회 자동 시작

    def on_frame(frame_count, events):
        for e in events:
            if e.action == "start_scenario" and e.value > 0:
                scenario_runner[0] = build_scenario_runner(low_amr)
                print("[test_lowamr] 시나리오 재시작: 전진+리프트업+회전+YawLock -> 복귀+리프트다운")

        runner = scenario_runner[0]
        if not runner.done:
            runner.update(frame_count)
        elif frame_count % 120 == 0:
            print("[test_lowamr] 시나리오 완료 (ENTER로 재시작 가능)")

    print("[test_lowamr] 시나리오 자동 실행 중 | ENTER: 재시작")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()