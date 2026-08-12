"""볶음 그릇 수동 파지 기록 전용 A0509 키보드 텔레오프.

공용 DoosanArmKeyboardTeleop은 수정하지 않고 그대로 상속한다. 자동 접근 뒤
수동 인계 구간에서만 ENTER 성공 표시를 추가하기 위해 이벤트 처리만 확장한다.
"""
from isaacgym import gymapi

from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop, _CONTINUOUS


class StirfryArmKeyboardTeleop(DoosanArmKeyboardTeleop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._success_requested = False
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "trace_success"
        )
        print(
            "[수동 궤적 기록] 그릇을 들어 올리는 데 성공하면 ENTER를 "
            "눌러 성공 시점을 기록하세요."
        )

    def handle_and_apply(self):
        """공용 조작 동작을 유지하면서 ENTER 성공 표시를 함께 처리한다."""
        for event in self.gym.query_viewer_action_events(self.viewer):
            action = event.action
            if action in _CONTINUOUS:
                (self.held.add if event.value > 0 else self.held.discard)(action)
                continue
            if event.value <= 0:
                continue
            if action == "mode_jsc":
                self.mode = "jsc"
                self.joint_target = self.arm.current_joints().copy()
                self.dirty = True
                print("[모드] JSC")
            elif action == "mode_tsc":
                self.mode = "tsc"
                self._sync_cart()
                self.dirty = True
                print("[모드] TSC")
            elif action == "mode_osc":
                self.mode = "osc"
                self._sync_cart()
                print("[모드] OSC")
            elif action == "joint_prev":
                self.sel_joint = (self.sel_joint - 1) % 6
                print(f"[관절] joint_{self.sel_joint + 1}")
            elif action == "joint_next":
                self.sel_joint = (self.sel_joint + 1) % 6
                print(f"[관절] joint_{self.sel_joint + 1}")
            elif action == "home":
                self.mode = "jsc"
                self.joint_target = self.HOME_Q.copy()
                self.dirty = True
                print("[홈] 천장 일자 → JSC")
            elif action == "trace_success":
                self._success_requested = True
                print("[수동 궤적 기록] ENTER 성공 표시를 받았습니다.")

        if self._apply_held():
            self.dirty = True
        self._apply_control()
        self._status()

    def consume_success_marker(self):
        """ENTER 성공 표시를 한 번만 반환한다."""
        requested = self._success_requested
        self._success_requested = False
        return requested
