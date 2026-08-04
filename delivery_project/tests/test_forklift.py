"""
tests/test_forklift.py — ForkliftAMR 시나리오 전용 테스트

다른 4개 테스트(conveyor/lowamr/indyarm/cardboardbox)와 동일하게 test_common.py의
공용 보일러플레이트(create_minimal_sim/create_viewer/load_asset/main_loop)를 쓴다.

시나리오는 main.py [SECTION 10-2]의 forklift_steps를 좌표/facing/lift_height/hold_seconds
그대로 복제했다 (제가 임의로 만든 좌표+hold_seconds=1.0 조합에서는 회전이 제대로 안 되는 문제가
있었고, main.py 원본 값 그대로 쓰니 정상 동작하는 것을 확인함 — 임의로 값을 바꾸지 말 것).
ANGLE_ARRIVE_THRESHOLD 등 ForkliftAMR 내부 상수도 건드리지 않는다(디버깅 중 오버라이드했던
적이 있었는데 그것도 문제를 일으켰을 수 있어 원복함).

main.py와의 유일한 의도적 차이:
  - forklift_steps 첫 스텝(INITIAL)의 wait_for=lambda ctx: ctx.forklift_start["start"] 제거
    (단독 테스트라 이 값을 True로 만들어줄 arm 러너가 없음)

실행하면 바로 시나리오가 1회 자동 시작된다.
    ENTER : 시나리오 재시작
"""

from isaacgym import gymapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_common import create_minimal_sim, create_viewer, load_asset, main_loop
from indy7_controller_v1 import StepContext
from forklift_amr_controller_v1 import ForkliftAMR, ForkliftMotionStep, ForkliftStepRunner


def build_scene():
    gym, sim, env, args = create_minimal_sim("ForkliftAMR Scenario Test")
    # main.py의 forklift 스폰 좌표(-3.2, -2.125, 0.3) 그대로 사용
    viewer = create_viewer(gym, sim, env, cam_pos=(-1.0, -4.5, 3.0), cam_target=(-2.5, -1.5, 0.3))

    forklift_asset = load_asset(
        gym, sim, "urdf/forklift/forklift_v1.urdf",
        fix_base_link=False, replace_cylinder_with_capsule=True,
    )
    forklift_handle = gym.create_actor(
        env, forklift_asset, gymapi.Transform(p=gymapi.Vec3(-3.2, -2.125, 0.3)), "forklift_test", -1, 1
    )
    forklift = ForkliftAMR(gym, sim, env, forklift_handle)

    gym.prepare_sim(sim)

    return gym, sim, env, viewer, forklift


def bind_keys(gym, viewer):
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, "start_scenario")


def build_scenario_runner(forklift):
    """
    main.py [SECTION 10-2] forklift_steps 그대로 복제 (좌표/facing/lift_height/hold_seconds 동일).
    유일한 차이는 첫 스텝의 wait_for(다른 컨트롤러 대기) 제거뿐이다.
    """
    steps = [
        ForkliftMotionStep(
            "INITIAL",
            target_x=-3.2, target_y=-2.125,
            axis_order='x',
            initial_facing='+x', direction1='FORWARD',
            mid_facing='+x', direction2='FORWARD',
            final_facing='+x',
            lift_height=-0.03,
            hold_seconds=0.1,
            # wait_for=lambda ctx: ctx.forklift_start["start"]  <- 단독 테스트라 제거
        ),
        ForkliftMotionStep(
            "MOVE_TO_SHELF_UNDER_AND_LIFT",
            target_x=-2.05, target_y=-2.125,
            axis_order='x',
            initial_facing='+x', direction1='FORWARD',
            mid_facing='+x', direction2='FORWARD',
            final_facing='+x',
            lift_height=0.15,
            hold_seconds=0.1,
        ),
        ForkliftMotionStep(
            "MOVE_SHELF",
            target_x=-2.5, target_y=-0.5,
            axis_order='x',
            initial_facing='+x', direction1='BACKWARD',
            mid_facing='+y', direction2='FORWARD',
            final_facing='-x',
            lift_height=-0.03,
            hold_seconds=0.1,
        ),
        ForkliftMotionStep(
            "MOVE_TO_SHELF_UNDER",
            target_x=-1.3, target_y=-1.0,
            axis_order='x',
            initial_facing='-x', direction1='BACKWARD',
            mid_facing='-y', direction2='FORWARD',
            final_facing='-y',
            lift_height=-0.03,
            hold_seconds=0.1,
        ),
    ]
    return ForkliftStepRunner(forklift, steps, context=StepContext(forklift=forklift))


def main():
    gym, sim, env, viewer, forklift = build_scene()
    bind_keys(gym, viewer)

    scenario_runner = [build_scenario_runner(forklift)]  # 실행하자마자 1회 자동 시작

    def on_frame(frame_count, events):
        for e in events:
            if e.action == "start_scenario" and e.value > 0:
                scenario_runner[0] = build_scenario_runner(forklift)
                print("[test_forklift] 시나리오 재시작")

        runner = scenario_runner[0]
        if not runner.done:
            runner.update(frame_count)
        elif frame_count % 120 == 0:
            print("[test_forklift] 시나리오 완료 (ENTER로 재시작 가능)")

    print("[test_forklift] main.py의 forklift_steps 그대로 재현 | ENTER: 재시작")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()