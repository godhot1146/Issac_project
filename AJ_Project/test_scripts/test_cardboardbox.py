"""
tests/test_cardboardbox.py — CardboardBoxManager 단독 테스트 (박스 1개, 공중 고정)

구성: 박스 1개. fix_base_link=True로 중력을 무시하고 공중에 고정해서 띄워두므로,
바닥에 떨어지거나 안착하는 것과 무관하게 접힘 스테이지를 전환 확인할 수 있다.

CardboardBoxManager는 자체 MotionStep/StepRunner가 없고 다른 컨트롤러의 콜백에서 호출당한다.
따라서 이 테스트의 "scenario 모드"는 진짜 StepRunner가 아니라, 그 역할을 흉내 낸
단순한 프레임 카운트 기반 시퀀서(BoxTestSequencer)를 스크립트 안에서 직접 구현한다.

[TAB] 모드 전환 (manual <-> scenario)

manual 모드 (원시 메서드 직접 호출):
    1 : apply_stage("unfolded_flat")
    2 : apply_stage("fold_sides")
    3 : apply_stage("close_fb_bottom")
    4 : apply_stage("close_rl_bottom")
    5 : apply_stage("close_top")
    K : unlock_joint
    
scenario 모드 (BoxTestSequencer — 콜백 기반 시퀀스 흉내):
    ENTER : fold_sides -> (1.5s 대기) -> close_fb_bottom -> (1.5s 대기) ->
            close_rl_bottom(+lock) -> (1.5s 대기) -> close_top
            순서로 자동 진행
"""

from isaacgym import gymapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controllers"))  # 제어 모듈 폴더

from test_common import create_minimal_sim, create_viewer, load_asset, main_loop, ModeToggle
from cardboard_box_manager import CardboardBoxManager, BoxStageRegistry


def build_registry():
    registry = BoxStageRegistry()
    registry.register("unfolded_flat", {
        "joint_right_to_front": 0.0, "joint_right_to_back": 0.0, "joint_front_to_left": 0.0,
        "joint_right_to_down": 0.0, "joint_left_to_down": 0.0, "joint_front_to_down": 0.0, "joint_back_to_down": 0.0,
        "joint_right_to_up": 0.0, "joint_left_to_up": 0.0, "joint_front_to_up": 0.0, "joint_back_to_up": 0.0,
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
    gym, sim, env, args = create_minimal_sim("CardboardBoxManager Unit Test")
    viewer = create_viewer(gym, sim, env, cam_pos=(1.5, -1.5, 1.5), cam_target=(0.0, 0.0, 1.0))

    registry = build_registry()

    # fix_base_link=True -> 중력 무시하고 스폰 위치에 고정(공중에 띄워진 상태 유지)
    box_asset = load_asset(gym, sim, "urdf/cardboard_box/v2/box_v2.urdf", fix_base_link=True, density=100.0)
    box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.0)), "box_test", -1, 0)
    box_manager = CardboardBoxManager(gym, sim, env, box_handle, stage_registry=registry, joint_speed=1.2)

    gym.prepare_sim(sim)

    box_manager.setup_tensors()

    return gym, sim, env, viewer, box_manager


def bind_keys(gym, viewer):
    for key, action in [
        (gymapi.KEY_1, "stage_unfolded"), (gymapi.KEY_2, "stage_fold_sides"),
        (gymapi.KEY_3, "stage_close_fb"), (gymapi.KEY_4, "stage_close_rl"),
        (gymapi.KEY_5, "stage_close_top"),
        (gymapi.KEY_K, "unlock_joint"),
        (gymapi.KEY_ENTER, "start_scenario"),
    ]:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)


class BoxTestSequencer:
    """
    ArmMotionStep의 on_start/on_complete가 하던 역할을 흉내 낸 소형 시퀀서.
    (name, action_fn, hold_frames) 리스트를 순서대로 실행한다.
    """
    def __init__(self, box_manager, steps):
        self.box_manager = box_manager
        self.steps = steps
        self.index = 0
        self.done = False
        self._entered = False
        self._hold_start_frame = None

    def update(self, frame_count):
        if self.done:
            return
        name, action_fn, hold_frames = self.steps[self.index]
        if not self._entered:
            print(f"[BoxTestSequencer] step -> {name}")
            action_fn(self.box_manager)
            self._entered = True
            self._hold_start_frame = frame_count
        if frame_count - self._hold_start_frame >= hold_frames:
            self.index += 1
            self._entered = False
            if self.index >= len(self.steps):
                self.done = True
                print("[BoxTestSequencer] 시퀀스 완료")


def build_scenario_sequencer(box_manager):
    steps = [
        ("FOLD_SIDES", lambda bm: bm.apply_stage("fold_sides"), 90),
        ("CLOSE_FB_BOTTOM", lambda bm: bm.apply_stage("close_fb_bottom"), 90),
        ("CLOSE_RL_BOTTOM", lambda bm: (bm.apply_stage("close_rl_bottom"),
                                         bm.lock_joint("joint_right_to_down"),
                                         bm.lock_joint("joint_left_to_down")), 90),
        ("CLOSE_TOP", lambda bm: bm.apply_stage("close_top"), 90),
    ]
    return BoxTestSequencer(box_manager, steps)


def main():
    gym, sim, env, viewer, box_manager = build_scene()
    bind_keys(gym, viewer)
    mode = ModeToggle(gym, viewer)

    scenario_seq = [None]

    def on_frame(frame_count, events):
        mode.handle_events(events)

        if mode.mode == "manual":
            for e in events:
                if e.value <= 0:
                    continue
                if e.action == "stage_unfolded":
                    box_manager.apply_stage("unfolded_flat")
                elif e.action == "stage_fold_sides":
                    box_manager.apply_stage("fold_sides")
                elif e.action == "stage_close_fb":
                    box_manager.apply_stage("close_fb_bottom")
                elif e.action == "stage_close_rl":
                    box_manager.apply_stage("close_rl_bottom")
                elif e.action == "stage_close_top":
                    box_manager.apply_stage("close_top")
                elif e.action == "unlock_joint":
                    box_manager.unlock_joint("joint_right_to_front")
                    box_manager.unlock_joint("joint_right_to_back")
                    box_manager.unlock_joint("joint_front_to_left")
                    box_manager.unlock_joint("joint_right_to_down")
                    box_manager.unlock_joint("joint_left_to_down")
                    box_manager.unlock_joint("joint_front_to_down")
                    box_manager.unlock_joint("joint_back_to_down")
                    box_manager.unlock_joint("joint_right_to_up")
                    box_manager.unlock_joint("joint_left_to_up")
                    box_manager.unlock_joint("joint_front_to_up")
                    box_manager.unlock_joint("joint_back_to_up")

        else:
            for e in events:
                if e.action == "start_scenario" and e.value > 0:
                    scenario_seq[0] = build_scenario_sequencer(box_manager)
                    print("[test_cardboardbox] 시나리오 시작: fold_sides -> close_fb -> close_rl(+lock) -> close_top")
            seq = scenario_seq[0]
            if seq is not None and not seq.done:
                seq.update(frame_count)

        # 매 프레임 공통 처리
        box_manager.update_joints()
        box_manager.update_platform_lock()
        box_manager.update_children()

    print("[test_cardboardbox] TAB: 모드 전환 | manual: 1-5/J/K | scenario: ENTER")
    main_loop(gym, sim, viewer, on_frame)


if __name__ == "__main__":
    main()