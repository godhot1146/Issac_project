"""
tests/test_indyarm_subarm.py — IndyArmWithSubArmController 단독 테스트

test_indyarm.py(기존 IndyArmController 테스트)를 그대로 베이스로 하되, sub_arm/cylinder가
추가된 새 URDF + IndyArmWithSubArmController로 교체한 버전이다. 달라진 점만 정리하면:

  - ee가 마지막 링크(sub_arm_part)가 아니라 'vacuum'이므로, print_status_report()의
    EE 좌표/attach 판정 기준점 등이 전부 vacuum 기준으로 바뀐다.
  - dof_count가 6 -> 8 (joint0~5 + sub_arm_joint + cylinder_joint)로 늘어났고,
    JOINT_MANUAL 모드는 원래부터 dof_count 기준으로 관절을 순회하므로 별도 수정 없이
    1/2(관절 선택)·3/4(각도 조절) 키로 sub_arm_joint/cylinder_joint도 바로 조작 가능하다.
    (JOINT_MANUAL에서 관절을 6번(joint5) 다음으로 한 번 더 넘기면 sub_arm_joint,
     그 다음이 cylinder_joint다.)
  - 추가로 sub_arm_joint만 즉시 까딱여보는 전용 키(N/B)를 붙여서, JOINT_MANUAL 모드로
    진입하지 않고도 manual 모드에서 바로 sub_arm 동작을 확인할 수 있게 했다.
  - scenario 시퀀스에 SUB_ARM_OPEN / SUB_ARM_CLOSE 스텝을 끼워 넣어서, ArmStepRunner가
    무수정인 채로 sub_arm_joint를 '관절 이동(JOINT)' 스텝으로 회전시키는 것을 확인한다.

[TAB] 모드 전환 (manual <-> scenario)
  scenario 모드로 들어가면 ArmStepRunner가 arm.set_automation_active(True)를 걸어서
  수동 키 입력이 전부 무시되는 것은 기존과 동일.

manual 모드:
    방향키 / Z,X          : EE(vacuum) 이동 (arm 자체 바인딩)
    T,Y,U,I,O,P           : EE(vacuum) 회전 6종 (arm 자체 바인딩)
    M                     : IK <-> JOINT_MANUAL 모드 전환
    1 / 2                 : (JOINT_MANUAL) 이전/다음 관절 선택 (joint0~5, sub_arm_joint,
                             cylinder_joint 총 8개 순환)
    3 / 4                 : (JOINT_MANUAL) 선택 관절 각도 감소/증가
    N / B                 : sub_arm_joint 즉시 -/+ 회전 (JOINT_MANUAL 진입 없이 테스트용)
    SPACE                 : print_status_report() (EE(vacuum)/전체 관절 상태 콘솔 출력)
    G / H                 : attach_nearest() / detach() (흡착 테스트)

scenario 모드 (ArmMotionStep + ArmStepRunner):
    ENTER : init_pose -> sub_arm 펼침 -> 더미 방향으로 이동+흡착 -> 원위치 복귀
            -> sub_arm 접음 순서의 시퀀스 시작
"""

from isaacgym import gymapi
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controllers"))  # 제어 모듈 폴더

from test_common import create_minimal_sim, create_viewer, load_asset, main_loop, ModeToggle
from indy7_controller_v1 import ArmMotionStep, ArmStepRunner, StepContext, flush_dof_targets
from indy7_box_controller import IndyArmWithSubArmController

# 실제 로봇/장착물 형상에 종속된 값이므로 프로젝트 좌표계에 맞게 조정 필요 (기존 test_indyarm.py와 동일 값)
INIT_POSE = [-2.0255, 0.2495, 0.9428, -0.0014, 1.9548, 0.4432]

# manual 모드 N/B 키로 sub_arm을 한 번 누를 때 움직일 각도
SUB_ARM_MANUAL_STEP = np.radians(5.0)

# scenario에서 사용할 sub_arm 펼침/접힘 각도
SUB_ARM_OPEN_DELTA = np.radians(30.0)


def build_scene(gym, sim, env, viewer):
    # sub_arm/cylinder가 추가된 새 URDF. 프로젝트 경로에 맞게 조정.
    arm_asset = load_asset(
        gym, sim,
        "urdf/indy_description/urdf_files/indy7_v3_vacuum_box.urdf",
        fix_base_link=True, density=100.0,
    )
    spawn_transform = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.5))
    arm = IndyArmWithSubArmController(
        gym, sim, env, viewer, arm_asset, spawn_transform=spawn_transform,
        actor_name="indy7_arm_subarm_test",
    )

    dummy_asset = load_asset(gym, sim, "urdf/warehouse/dummy.urdf", fix_base_link=False, density=100.0)
    dummy_pos = gymapi.Vec3(0.3, 0.0, 0.55)
    dummy_handle = gym.create_actor(env, dummy_asset, gymapi.Transform(p=dummy_pos), "attach_test_dummy", -1, 0)

    gym.prepare_sim(sim)
    arm.setup_tensors()
    arm.register_attachable(dummy_handle)
    # 6-DOF(팔)만 넘겨도 register_joint_pose가 현재 sub_arm/cylinder 목표값을 붙여서 8-DOF로 저장한다.
    arm.register_joint_pose("init_pose", INIT_POSE)

    print(f"[test_indyarm_subarm] ee_name = {arm.ee_name}  "
          f"(마지막 링크가 아니라 vacuum이어야 정상)")
    print(f"[test_indyarm_subarm] dof_count = {arm.dof_count}, dof_names = {arm.dof_names}")
    print(f"[test_indyarm_subarm] SUB_ARM_DOF_IDX = {arm.SUB_ARM_DOF_IDX}, "
          f"CYLINDER_DOF_IDX = {arm.CYLINDER_DOF_IDX}")

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
        (gymapi.KEY_N, "sub_arm_dec"),   # sub_arm_joint 즉시 -SUB_ARM_MANUAL_STEP
        (gymapi.KEY_B, "sub_arm_inc"),   # sub_arm_joint 즉시 +SUB_ARM_MANUAL_STEP
        (gymapi.KEY_ENTER, "start_scenario"),
    ]:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)


def build_scenario_runner(arm, dummy_pos):
    """
    dummy_pos: build_scene()에서 스폰한 attach 테스트용 더미의 '스폰 시점' 월드 좌표.
    delta_value 추정/보정 방법은 기존 test_indyarm.py의 build_scenario_runner 주석과 동일하다
    (manual 모드로 조그 -> SPACE로 EE 좌표 확인 -> 역산).

    sub_arm은 IK 대상이 아니라 JOINT 스텝으로 넣는다 (vacuum=ee 기준으로 sub_arm_joint는
    자식 관절이라 IK가 건드리지 않으므로, 펼침/접힘은 명시적으로 JOINT 스텝을 써야 함).
    """
    ctx = StepContext(arm=arm)

    def _attach(ctx):
        ctx.arm.attach_nearest(distance_threshold=0.3)

    def _detach(ctx):
        ctx.arm.detach()

    steps = [
        ArmMotionStep("MOVE_TO_INIT", "joint", target="init_pose"),
        ArmMotionStep("SUB_ARM_OPEN", "joint",
                      target=arm.sub_arm_delta_target(SUB_ARM_OPEN_DELTA),
                      stable_frames=10),
        ArmMotionStep("APPROACH_XY", "cartesian_delta", delta_axis=0, delta_value=0.2,
                      linear_speed=0.3, on_complete=[]),
        ArmMotionStep("APPROACH_Z", "cartesian_delta", delta_axis=2, delta_value=-0.1,
                      linear_speed=0.3, on_complete=[_attach]),
        ArmMotionStep("RETREAT_Z", "cartesian_delta", delta_axis=2, delta_value=0.1,
                      linear_speed=0.3, on_complete=[]),
        ArmMotionStep("SUB_ARM_CLOSE", "joint",
                      target=arm.sub_arm_abs_target(0.0),
                      stable_frames=10),
        ArmMotionStep("BACK_TO_INIT", "joint", target="init_pose", on_complete=[_detach]),
    ]
    return ArmStepRunner(arm, steps, context=ctx)


def main():
    gym, sim, env, args = create_minimal_sim("IndyArmWithSubArmController Unit Test")
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
                elif e.action == "sub_arm_dec":
                    # JOINT_MANUAL 모드로 안 들어가도 sub_arm_joint만 바로 까딱여보는 테스트용
                    arm.move_sub_arm_by(-SUB_ARM_MANUAL_STEP)
                elif e.action == "sub_arm_inc":
                    arm.move_sub_arm_by(+SUB_ARM_MANUAL_STEP)
        else:
            for e in events:
                if e.action == "start_scenario" and e.value > 0:
                    scenario_runner[0] = build_scenario_runner(arm, dummy_pos)
                    print("[test_indyarm_subarm] 시나리오 시작: "
                          "init -> sub_arm 펼침 -> 더미로 이동+흡착 -> init 복귀 -> sub_arm 접음+해제")
            runner = scenario_runner[0]
            if runner is not None and not runner.done:
                runner.update(frame_count)
            elif runner is not None and runner.done and frame_count % 120 == 0:
                print("[test_indyarm_subarm] 시나리오 완료 (ENTER로 재시작 가능)")

        # 매 프레임 공통 처리 (수동/자동 모드 공통)
        arm.step()
        arm.update_attachment()
        arm.draw_target_marker()
        arm.draw_ee_marker(color=(0.0, 1.0, 0.0))

        flush_dof_targets(gym, sim, [arm])

    print("[test_indyarm_subarm] TAB: 모드 전환 | "
          "manual: 방향키/Z,X/T,Y,U,I,O,P/M/1-4/N,B/G/H | scenario: ENTER")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()