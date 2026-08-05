"""
tests/test_conveyor.py — ConveyorBelt 단독 테스트

구성: 컨베이어 1개 + 벨트에 태울 박스(또는 dummy) 1개.
다른 컨트롤러(팔/AMR)는 전혀 필요 없다 (ConveyorBelt는 register_item()만 받으면 됨).

[TAB] 모드 전환 (manual <-> scenario)

manual 모드 (원시 메서드 직접 호출):
    SPACE : move_forward()  (거리 제한 없이 가동)
    B     : move_backward()  (S는 Isaac Gym 뷰어 기본 프리카메라 키와 겹쳐서 B로 뺌)
    X     : stop()
    UP    : set_speed(현재+0.1)
    DOWN  : set_speed(현재-0.1, 최소 0.05)
    L     : 박스를 초기 스폰 위치(box_start_pos)로 텔레포트 + 안착/거리 추적 리셋
            (벨트 밖으로 나가서 unregister된 상태였다면 register_item도 자동 재실행)

scenario 모드 (ConveyorMotionStep + ConveyorStepRunner):
    ENTER : FORWARD 1.0m(hold 1s) -> BACKWARD 1.0m(hold 1s) 왕복 시퀀스 새로 시작
"""

import glob
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controllers"))  # 제어 모듈 폴더

from test_common import (
    ASSET_ROOT, create_minimal_sim, create_viewer, load_asset, main_loop,
    ModeToggle,
)
from conveyor_belt import ConveyorBelt, ConveyorMotionStep, ConveyorStepRunner
from cardboard_box_manager import CardboardBoxManager, BoxStageRegistry
from indy7_controller_v1 import StepContext

# 박스를 스폰 시점에 어떤 스테이지로 강제할지. BoxStageRegistry에 등록된 이름 중 선택.
# "fold_sides"(옆면만 접힘) / "close_top"(완전히 닫힌 상자) 등으로 바꿔서 테스트 가능.
INITIAL_BOX_STAGE = "close_top"

# 벨트 실제 물리 길이(m) — setup_tensors(belt_length=...)와 이탈 감지 판정에 공용으로 사용.
BELT_LENGTH = 3.5


def build_registry():
    """
    CardboardBoxManager_handoff.md / main.py Section 5와 동일한 스테이지 체인.
    close_top 등 상위 스테이지는 base로 하위 스테이지를 누적 상속하므로, 목표 스테이지가
    무엇이든 체인 전체를 등록해둬야 한다.
    """
    registry = BoxStageRegistry()
    registry.register("unfolded_flat", {
        "joint_right_to_front": 10.0, "joint_right_to_back": 10.0, "joint_front_to_left": -10.0,
    })
    registry.register("fold_sides", {
        "joint_right_to_front": 90.0, "joint_right_to_back": 90.0, "joint_front_to_left": -90.0,
    })
    registry.register(
        "close_fb_bottom",
        joint_targets_deg={"joint_front_to_down": -90.0, "joint_back_to_down": 90.0},
        base="fold_sides",
    )
    registry.register(
        "close_rl_bottom",
        joint_targets_deg={"joint_right_to_down": -90.0, "joint_left_to_down": 90.0},
        base="close_fb_bottom",
        lock_joints=["joint_right_to_down", "joint_left_to_down",
                     "joint_front_to_down", "joint_back_to_down"],
    )
    registry.register(
        "close_top",
        joint_targets_deg={
            "joint_right_to_up": 90.0, "joint_left_to_up": -90.0,
            "joint_front_to_up": 90.0, "joint_back_to_up": -90.0,
        },
        base="close_rl_bottom",
        lock_joints=["joint_right_to_up", "joint_left_to_up",
                     "joint_front_to_up", "joint_back_to_up"],
    )
    return registry


def build_scene():
    gym, sim, env, args = create_minimal_sim("ConveyorBelt Unit Test")
    viewer = create_viewer(gym, sim, env, cam_pos=(2.5, -2.5, 2.0), cam_target=(0.0, 0.0, 0.3))

    conveyor_asset = load_asset(gym, sim, "urdf/conveyor/v2/conveyor_v2.urdf", fix_base_link=True, density=1000.0)
    conveyor_pos = gymapi.Vec3(0.0, 0.0, 0.05)
    conveyor_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0))
    conveyor_transform = gymapi.Transform(p=conveyor_pos, r=conveyor_rot)
    conveyor_handle = gym.create_actor(
        env, conveyor_asset, conveyor_transform, "conveyor_test", -1, 0
    )

    # 벨트 위에 태울 물체 — 실제 에셋 치수/벨트 시작점 좌표는 프로젝트에 맞게 조정
    box_asset = load_asset(gym, sim, "urdf/cardboard_box/v2/box_v2.urdf", fix_base_link=False, density=100.0)
    box_start_pos = gymapi.Vec3(0.0, 0.15, 1.2)
    box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=box_start_pos), "conveyor_test_box", -1, 0)

    # CardboardBoxManager 생성은 prepare_sim 이전(생성자 호출) 규칙을 따름
    registry = build_registry()
    box_manager = CardboardBoxManager(gym, sim, env, box_handle, stage_registry=registry, joint_speed=1.2)

    frame_paths = sorted(glob.glob(os.path.join(ASSET_ROOT, "urdf/conveyor/v2/anim/*.png")))
    conveyor = ConveyorBelt(
        gym, sim, env, conveyor_handle,
        belt_link_name="belt_link",
        texture_frame_paths=frame_paths,
        speed=0.5,
    )

    gym.prepare_sim(sim)

    _root_state_raw = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(_root_state_raw)

    box_manager.setup_tensors()
    box_manager.set_initial_spawn_stage(INITIAL_BOX_STAGE)

    conveyor.setup_tensors(belt_length=BELT_LENGTH)
    ok = conveyor.register_item(box_handle)

    return gym, sim, env, viewer, conveyor, box_manager, box_handle, box_start_pos, conveyor_transform, root_states


def bind_keys(gym, viewer):
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "move_forward")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "move_backward")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "stop")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "speed_up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "speed_down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_L, "reset_landing")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, "start_scenario")


def check_off_belt(gym, env, box_handle, conveyor_transform, belt_length, margin=0.1):
    """
    박스의 월드 좌표를 컨베이어 로컬 좌표로 변환해, 진행축(로컬 y)이 벨트 물리 길이의
    절반(+margin)을 넘었는지 확인한다. ConveyorBelt 자체엔 이런 이탈 감지가 없어서
    (문서 3장 참고 — moved_distance는 '이번 run에서 이동한 거리'일 뿐, 벨트 절대 범위와
    무관) 테스트 스크립트에서 직접 계산한다. 컨베이어는 fix_base_link=True라 스폰 시점
    transform이 절대 안 바뀌므로, 그 transform의 역변환만으로 충분하다(매 프레임 벨트
    포즈를 다시 쿼리할 필요 없음).
    """
    body_states = gym.get_actor_rigid_body_states(env, box_handle, gymapi.STATE_POS)
    world_pos = body_states['pose']['p'][0]
    world_vec = gymapi.Vec3(float(world_pos['x']), float(world_pos['y']), float(world_pos['z']))
    local_vec = conveyor_transform.inverse().transform_point(world_vec)
    return abs(local_vec.y) > (belt_length / 2.0 + margin), local_vec.y


def reset_box_position(gym, sim, env, box_handle, root_states, target_pos):
    """
    박스(다물체 articulated 액터)의 루트 링크 위치/회전/속도만 텐서 API로 리셋한다.
    관절(DOF) 각도는 전혀 건드리지 않으므로 현재 접힘 형상(잠긴 스테이지)은 그대로 유지된 채,
    나머지 링크들은 articulation FK로 새 루트 위치를 따라 자동 재배치된다.

    이전에 gym.set_actor_rigid_body_states로 링크들을 하나하나 직접 옮겼을 때
    "[Warning] Incomplete articulation start index in function setRigidBodyStates"가
    떴던 건, GPU 파이프라인에서 articulated 액터의 링크 상태를 그 방식으로 직접 쓰는 게
    지원 범위 밖이기 때문. 루트 상태 텐서(set_actor_root_state_tensor_indexed)로 바꾸면
    정상적인 방식이라 경고가 안 뜬다.

    root_states: build_scene()에서 gym.acquire_actor_root_state_tensor(sim)로 미리 획득해
    gymtorch.wrap_tensor()로 감싸둔 텐서를 그대로 재사용(매 호출마다 새로 acquire하지 않음).
    """
    actor_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)

    root_states[actor_idx, 0] = target_pos.x
    root_states[actor_idx, 1] = target_pos.y
    root_states[actor_idx, 2] = target_pos.z
    root_states[actor_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=root_states.device)  # identity quat
    root_states[actor_idx, 7:13] = 0.0  # 선속도/각속도 0

    actor_indices = torch.tensor([actor_idx], dtype=torch.int32, device=root_states.device)
    gym.set_actor_root_state_tensor_indexed(
        sim, gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(actor_indices), 1,
    )


def build_scenario_runner(conveyor):
    steps = [
        ConveyorMotionStep("FWD", distance=1.0, direction="FORWARD", hold_seconds=1.0),
        ConveyorMotionStep("BWD", distance=1.0, direction="BACKWARD", hold_seconds=1.0),
    ]
    return ConveyorStepRunner(conveyor, steps, context=StepContext(conveyor=conveyor))


def main():
    gym, sim, env, viewer, conveyor, box_manager, box_handle, box_start_pos, conveyor_transform, root_states = build_scene()
    bind_keys(gym, viewer)
    mode = ModeToggle(gym, viewer)

    current_speed = 0.5
    scenario_runner = [None]  # 클로저에서 재할당하기 위한 1칸짜리 컨테이너
    item_state = {"on_belt": True}  # 자동 이탈 감지 상태 (중복 unregister/로그 방지용)

    def on_frame(frame_count, events):
        nonlocal current_speed
        mode.handle_events(events)

        if mode.mode == "manual":
            for e in events:
                if e.value <= 0:
                    continue
                if e.action == "move_forward":
                    conveyor.move_forward()
                elif e.action == "move_backward":
                    conveyor.move_backward()
                elif e.action == "stop":
                    conveyor.stop()
                elif e.action == "speed_up":
                    current_speed += 0.1
                    conveyor.set_speed(current_speed)
                    print(f"[speed] {current_speed:.2f} m/s")
                elif e.action == "speed_down":
                    current_speed = max(0.05, current_speed - 0.1)
                    conveyor.set_speed(current_speed)
                    print(f"[speed] {current_speed:.2f} m/s")
                elif e.action == "reset_landing":
                    reset_box_position(gym, sim, env, box_handle, root_states, box_start_pos)
                    conveyor.reset_item_landing(box_handle)
                    if not item_state["on_belt"]:
                        conveyor.register_item(box_handle)
                        item_state["on_belt"] = True
                    print("[test_conveyor] 박스 위치 초기화 + landing 상태 리셋")
            conveyor.update(frame_count)

        else:  # scenario
            for e in events:
                if e.action == "start_scenario" and e.value > 0:
                    scenario_runner[0] = build_scenario_runner(conveyor)
                    print("[test_conveyor] 시나리오 시작: FWD 1.0m -> BWD 1.0m")
            runner = scenario_runner[0]
            if runner is not None and not runner.done:
                runner.update(frame_count)
            elif runner is not None and runner.done and frame_count % 120 == 0:
                print("[test_conveyor] 시나리오 완료 (ENTER로 재시작 가능)")

        # 박스 관절/오프셋 잠금 후처리 — CardboardBoxManager는 매 프레임 반드시 호출해야 함
        # (main.py Section 12-6과 동일)
        box_manager.update_joints()
        box_manager.update_platform_lock()
        box_manager.update_children()

        # 벨트 이탈 자동 감지 (ConveyorBelt 자체엔 없는 기능 — 여기서 직접 계산)
        if item_state["on_belt"]:
            off_belt, local_y = check_off_belt(gym, env, box_handle, conveyor_transform, BELT_LENGTH)
            if off_belt:
                conveyor.unregister_item(box_handle)
                item_state["on_belt"] = False
                print(f"[test_conveyor] 박스가 벨트를 벗어남 감지 (local_y={local_y:.3f}m, "
                      f"벨트 절반={BELT_LENGTH / 2:.2f}m) -> unregister_item 자동 호출, "
                      f"이후 자연 물리로 낙하/정지")

    print("[test_conveyor] TAB: 모드 전환 | manual: SPACE/B/X/UP/DOWN/L | scenario: ENTER")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()