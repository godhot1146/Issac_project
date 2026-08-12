"""볶음 그릇 수동 파지 기록 전용 A0509 키보드 텔레오프.

공용 DoosanArmKeyboardTeleop은 수정하지 않고 그대로 상속한다. 자동 접근 뒤
수동 인계 구간에서만 ENTER 성공 표시를 추가하기 위해 이벤트 처리만 확장한다.
"""
from isaacgym import gymapi

from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop, _CONTINUOUS


class StirfryArmKeyboardTeleop(DoosanArmKeyboardTeleop):
    def __init__(self, *args, **kwargs):
        self.gravity_control = kwargs.pop("gravity_control", None)
        if self.gravity_control is None:
            raise ValueError("볶음 전용 텔레오프에는 gravity_control이 필요합니다.")
        super().__init__(*args, **kwargs)
        self._success_requested = False
        self._sync_tsc_without_ik()
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "trace_success"
        )
        print(
            "[수동 궤적 기록] 그릇을 들어 올리는 데 성공하면 ENTER를 "
            "눌러 성공 시점을 기록하세요."
        )

    def _print_help(self):
        print("""
===== 볶음 파지 안전 키보드 제어 (중력보상 유지) =====
 [모드] 1:JSC  2:TSC 안전 재동기화  3:OSC 비활성화
 [좌표이동·TSC] W/S:X±  A/D:Y±  Q/E:Z±
 [자세회전·TSC] T/G:pitch±  F/H:roll±  C/V:yaw±
 [관절·JSC] J/L:관절선택  U/O:각도±
 [기록] 그릇을 안정적으로 들면 ENTER
 [주의] R은 홈 복귀이므로 파지 중 누르지 마세요.
====================================================""")

    def _sync_tsc_without_ik(self):
        """현재 관절 자세를 그대로 고정하고 TSC 기준만 다시 맞춘다."""
        current_joints = self.arm.current_joints().copy()
        self.mode = "tsc"
        self.joint_target = current_joints.copy()
        self.arm._last_q = current_joints.copy()
        self.gravity_control.capture_current_target()
        self.gravity_control.update()
        self._sync_cart()
        self.dirty = False

    def _apply_control(self):
        """JSC/TSC에서도 자동 접근과 같은 중력보상 위치제어를 유지한다."""
        if self.mode == "jsc":
            if self.dirty:
                self.gravity_control.command(self.joint_target)
                self.dirty = False
        elif self.mode == "tsc":
            if self.dirty:
                target_joints, _, _ = self.arm.solve_ik(
                    self.cart_target,
                    target_R=self.ori_R,
                    seed_6=self.arm._last_q,
                )
                self.arm._last_q = target_joints.copy()
                self.gravity_control.command(target_joints)
                self.dirty = False
        else:
            super()._apply_control()

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
                self._sync_tsc_without_ik()
                print("[모드] TSC (현재 자세에서 안전 재동기화)")
            elif action == "mode_osc":
                print(
                    "[안전] 볶음 파지 중 OSC 전환은 비활성화되어 있습니다. "
                    "중력보상 TSC를 유지합니다."
                )
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
