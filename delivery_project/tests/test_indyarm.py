"""
tests/test_indyarm.py — IndyArmController 단독 테스트

구성: 팔 1개 + 흡착 테스트용 더미(박스) 1개.

IndyArmController는 생성자 안에서 방향키/Z,X/T,Y,U,I,O,P를 이미 자체적으로 바인딩하므로
(§ IndyArmController_인수인계문서.md 2.6 참고), manual 모드에서는 그냥
arm.process_keyboard_input(events)만 매 프레임 호출해주면 그 키들이 그대로 동작한다.
이 스크립트가 추가로 subscribe하는 키는 main Section 11 중 팔 관련 키(SPACE, M, 1~4)와,
흡착 테스트용 커스텀 키(G/H) 뿐이다.

[TAB] 모드 전환 (manual <-> scenario)
  주의: scenario 모드로 들어가면 ArmStepRunner가 arm.set_automation_active(True)를 걸어서
  수동 키 입력(방향키 등)이 자동으로 무시된다 (§ 마스터 문서 4.3 주의사항과 동일한 현상을
  단독 테스트에서도 그대로 재현/확인할 수 있음).

manual 모드:
    방향키 / Z,X          : EE 이동 (arm 자체 바인딩)
    T,Y,U,I,O,P           : EE 회전 6종 (arm 자체 바인딩)
    M                     : IK <-> JOINT_MANUAL 모드 전환
    1 / 2                 : (JOINT_MANUAL) 이전/다음 관절 선택
    3 / 4                 : (JOINT_MANUAL) 선택 관절 각도 감소/증가
    SPACE                 : print_status_report() (현재 EE/관절 상태 콘솔 출력)
    G / H                 : attach_nearest() / detach()  (흡착 테스트)

scenario 모드 (ArmMotionStep + ArmStepRunner):
    ENTER : init_pose -> 더미 방향으로 이동+흡착 -> 원위치 복귀 3스텝 시퀀스 시작
"""

from isaacgym import gymapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_common import create_minimal_sim, create_viewer, load_asset, main_loop, ModeToggle
from indy7_controller_v1 import (
    IndyArmController, ArmMotionStep, ArmStepRunner, StepContext, flush_dof_targets,
)

# 실제 로봇/장착물 형상에 종속된 값이므로 프로젝트 좌표계에 맞게 조정 필요
INIT_POSE = [-2.0255, 0.2495, 0.9428, -0.0014, 1.9548, 0.4432]


def build_scene(gym, sim, env, viewer):
    arm_asset = load_asset(gym, sim, "urdf/indy_description/urdf_files/indy7_v3_vacuum.urdf", fix_base_link=True, density=100.0)
    spawn_transform = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.5))
    arm = IndyArmController(gym, sim, env, viewer, arm_asset, spawn_transform=spawn_transform,
                             actor_name="indy7_arm_test")

    dummy_asset = load_asset(gym, sim, "urdf/warehouse/dummy.urdf", fix_base_link=False, density=100.0)
    dummy_pos = gymapi.Vec3(0.3, 0.0, 0.55)
    dummy_handle = gym.create_actor(env, dummy_asset, gymapi.Transform(p=dummy_pos), "attach_test_dummy", -1, 0)

    gym.prepare_sim(sim)
    arm.setup_tensors()
    arm.register_attachable(dummy_handle)
    arm.register_joint_pose("init_pose", INIT_POSE)

    return arm, dummy_handle, dummy_pos


def bind_keys(gym, viewer):
    for key, action in [
        (gymapi.KEY_SPACE, "start_process"),
        (gymapi.KEY_M, "toggle_manual_mode"),
        (gymapi.KEY_1, "select_prev_joint"),
        (gymapi.KEY_2, "select_next_joint"),
        (gymapi.KEY_3, "decrease_joint_angle"),
        (gymapi.KEY_4, "increase_joint_angle"),
        (gymapi.KEY_G, "attach_nearest"),
        (gymapi.KEY_H, "detach"),
        (gymapi.KEY_ENTER, "start_scenario"),
    ]:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)


def build_scenario_runner(arm, dummy_pos):
    """
    dummy_pos: build_scene()에서 스폰한 attach 테스트용 더미의 '스폰 시점' 월드 좌표.
    init_pose에서의 실제 EE 위치를 모르는 상태이므로 절대좌표(cartesian_abs) 대신,
    main에서 실제로 쓰이는 cartesian_delta(현재 EE 기준 상대이동)로 접근한다.

    아래 delta_value(0.2, -0.1 등)는 "arm 스폰 위치 (0,0,0.5) + init_pose 관절각"일 때
    EE가 대략 어디 있을지 추정한 placeholder다. 실제로는:
      1) manual 모드로 진입해 방향키로 더미 근처까지 직접 조그(jog)해보고
      2) SPACE(print_status_report)로 EE 월드 좌표를 확인한 뒤
      3) init_pose 시점 EE 좌표와의 차이를 delta_value로 역산
    해서 채워 넣는 걸 권장한다. distance_threshold를 넉넉히(0.3) 잡아둔 것도
    이 추정 오차를 흡수하기 위함이다.
    """
    ctx = StepContext(arm=arm)

    def _attach(ctx):
        ctx.arm.attach_nearest(distance_threshold=0.3)

    def _detach(ctx):
        ctx.arm.detach()

    steps = [
        ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
        ArmMotionStep("APPROACH_XY", "cartesian_delta", delta_axis=0, delta_value=0.2,
                      linear_speed=0.3, on_complete=[]),
        ArmMotionStep("APPROACH_Z", "cartesian_delta", delta_axis=2, delta_value=-0.1,
                      linear_speed=0.3, on_complete=[_attach]),
        ArmMotionStep("RETREAT_Z", "cartesian_delta", delta_axis=2, delta_value=0.1,
                      linear_speed=0.3, on_complete=[]),
        ArmMotionStep("BACK_TO_INIT", "joint", target="init_pose", on_complete=[_detach]),
    ]
    return ArmStepRunner(arm, steps, context=ctx)


def main():
    gym, sim, env, args = create_minimal_sim("IndyArmController Unit Test")
    viewer = create_viewer(gym, sim, env, cam_pos=(1.5, -1.5, 1.5), cam_target=(0.0, 0.0, 0.5))

    arm, dummy_handle, dummy_pos = build_scene(gym, sim, env, viewer)
    bind_keys(gym, viewer)
    mode = ModeToggle(gym, viewer)

    scenario_runner = [None]

    def on_frame(frame_count, events):
        mode.handle_events(events)

        if mode.mode == "manual":
            arm.process_keyboard_input(events)
            for e in events:
                if e.value <= 0:
                    continue
                if e.action == "attach_nearest":
                    arm.attach_nearest(distance_threshold=0.3)
                elif e.action == "detach":
                    arm.detach()
        else:
            for e in events:
                if e.action == "start_scenario" and e.value > 0:
                    scenario_runner[0] = build_scenario_runner(arm, dummy_pos)
                    print("[test_indyarm] 시나리오 시작: init -> 더미로 이동+흡착 -> init 복귀+해제")
            runner = scenario_runner[0]
            if runner is not None and not runner.done:
                runner.update(frame_count)
            elif runner is not None and runner.done and frame_count % 120 == 0:
                print("[test_indyarm] 시나리오 완료 (ENTER로 재시작 가능)")

        # 매 프레임 공통 처리 (수동/자동 모드 공통, main Section 12-3와 동일)
        arm.step()
        arm.update_attachment()
        arm.draw_target_marker()
        arm.draw_ee_marker(color=(0.0, 1.0, 0.0))

        flush_dof_targets(gym, sim, [arm])

    print("[test_indyarm] TAB: 모드 전환 | manual: 방향키/Z,X/T,Y,U,I,O,P/M/1-4/G/H | scenario: ENTER")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()