"""
keyboard_teleop.py — 두산 A0509 키보드 텔레오프 (JSC / TSC / OSC 공용)

런파일들이 공유하는 실시간 키보드 조작기.
  - 키를 '꾹 누르면 계속' 동작 (press/release 상태 추적)
  - TSC: 평행이동 중에도 손끝 자세(그리퍼 절대각) 유지 + 자세 회전 키
  - 홈(R) = 천장 향해 일자(전관절 0)

사용:
    from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop
    teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)   # 키 등록 + 초기 홈자세
    while not gym.query_viewer_has_closed(viewer):
        teleop.handle_and_apply()          # 이벤트 처리 + 연속동작 + 제어 인가
        gym.simulate(sim); gym.fetch_results(sim, True)
        ... (파일별 draw) ...

키 맵:
  [모드] 1:JSC  2:TSC  3:OSC
  [좌표이동·TSC/OSC] W/S:X±  A/D:Y±  Q/E:Z±   (자세 유지)
  [자세회전·TSC]     T/G:pitch±  F/H:roll±  C/V:yaw±
  [관절·JSC]         J/L:관절선택   U/O:각도±
  [공통] R:홈(천장 일자)
"""
import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R

# 꾹 누르면 계속 동작하는 액션(연속) vs 누를 때 한 번(단발)
_CONTINUOUS = {"x+", "x-", "y+", "y-", "z+", "z-",
               "roll+", "roll-", "pitch+", "pitch-", "yaw+", "yaw-",
               "joint+", "joint-"}
_AXIS = {"x": 0, "y": 1, "z": 2}
_ORI_AXIS = {"roll": "x", "pitch": "y", "yaw": "z"}


class DoosanArmKeyboardTeleop:
    HOME_Q = np.zeros(6, dtype=np.float32)   # 천장 향해 일자(전관절 0)

    def __init__(self, gym, viewer, arm, base_z=0.0,
                 cart_step=0.004, joint_step=0.01, ori_step_deg=1.5,
                 settle=30, home_on_start=True):
        self.gym, self.viewer, self.arm = gym, viewer, arm
        self.base_z = base_z
        self.cart_step = cart_step
        self.joint_step = joint_step
        self.ori_step = np.deg2rad(ori_step_deg)

        self.keymap = {
            gymapi.KEY_1: "mode_jsc", gymapi.KEY_2: "mode_tsc", gymapi.KEY_3: "mode_osc",
            gymapi.KEY_W: "x+", gymapi.KEY_S: "x-",
            gymapi.KEY_A: "y+", gymapi.KEY_D: "y-",
            gymapi.KEY_Q: "z+", gymapi.KEY_E: "z-",
            gymapi.KEY_T: "pitch+", gymapi.KEY_G: "pitch-",
            gymapi.KEY_F: "roll+",  gymapi.KEY_H: "roll-",
            gymapi.KEY_C: "yaw+",   gymapi.KEY_V: "yaw-",
            gymapi.KEY_J: "joint_prev", gymapi.KEY_L: "joint_next",
            gymapi.KEY_U: "joint+",     gymapi.KEY_O: "joint-",
            gymapi.KEY_R: "home",
        }
        for k, a in self.keymap.items():
            gym.subscribe_viewer_keyboard_event(viewer, k, a)

        self.held = set()
        self.mode = "tsc"
        self.sel_joint = 0
        self._n = 0

        if home_on_start:
            # 일반 수동 실행의 초기 자세: 천장 향해 일자
            self.joint_target = self.HOME_Q.copy()
            self.dirty = True
            arm.go_joints(self.HOME_Q)
            for _ in range(settle):
                gym.simulate(arm.sim); gym.fetch_results(arm.sim, True)
        else:
            # 자동 접근에서 인계받을 때는 홈으로 되돌리지 않고 현재 자세와
            # 현재 position drive 목표를 그대로 유지한다.
            self.joint_target = arm.current_joints().copy()
            self.dirty = False

        # TSC 목표(위치+자세)를 현재 손끝에 동기화(튐 방지)
        p, Rm = arm.current_pose()
        self.cart_target = p.astype(np.float32)
        self.ori_R = Rm
        self._print_help()
        if not home_on_start:
            print(
                "[키보드 인계] 현재 그리퍼 자세를 유지한 채 TSC로 시작합니다. "
                "2번을 누르면 언제든 현재 자세에 다시 동기화됩니다."
            )

    # ---------------------------------------------------------------
    def _print_help(self):
        print("""
===== 두산 A0509 키보드 제어 (꾹 누르면 연속) =====
 [모드] 1:JSC(관절)  2:TSC(좌표+자세IK)  3:OSC(좌표+동역학)
 [좌표이동·TSC/OSC] W/S:X±  A/D:Y±  Q/E:Z±   (자세 유지)
 [자세회전·TSC]     T/G:pitch±  F/H:roll±  C/V:yaw±
 [관절·JSC]         J/L:관절선택   U/O:각도±
 [공통] R:홈(천장 일자)     (창 닫기: 종료)
==================================================""")

    def _sync_cart(self):
        p, Rm = self.arm.current_pose()
        self.cart_target = p.astype(np.float32)
        self.ori_R = Rm

    # ---------------------------------------------------------------
    def handle_and_apply(self):
        """이벤트 처리 → 연속 동작 적용 → 모드별 제어 인가. 매 프레임 호출."""
        for e in self.gym.query_viewer_action_events(self.viewer):
            a = e.action
            if a in _CONTINUOUS:                    # 연속: 눌림/뗌 상태만 추적
                (self.held.add if e.value > 0 else self.held.discard)(a)
                continue
            if e.value <= 0:                        # 단발: 눌릴 때만
                continue
            if a == "mode_jsc":
                self.mode = "jsc"; self.joint_target = self.arm.current_joints().copy()
                self.dirty = True; print("[모드] JSC")
            elif a == "mode_tsc":
                self.mode = "tsc"; self._sync_cart(); self.dirty = True; print("[모드] TSC")
            elif a == "mode_osc":
                self.mode = "osc"; self._sync_cart(); print("[모드] OSC")
            elif a == "joint_prev":
                self.sel_joint = (self.sel_joint - 1) % 6; print(f"[관절] joint_{self.sel_joint+1}")
            elif a == "joint_next":
                self.sel_joint = (self.sel_joint + 1) % 6; print(f"[관절] joint_{self.sel_joint+1}")
            elif a == "home":
                self.mode = "jsc"; self.joint_target = self.HOME_Q.copy()
                self.dirty = True; print("[홈] 천장 일자 → JSC")

        if self._apply_held():
            self.dirty = True
        self._apply_control()
        self._status()

    def _apply_held(self):
        """눌려있는 연속 키만큼 목표를 이동. 뭔가 움직였으면 True."""
        if not self.held:
            return False
        for a in self.held:
            if a in ("x+", "x-", "y+", "y-", "z+", "z-"):
                self.cart_target[_AXIS[a[0]]] += self.cart_step if a[1] == "+" else -self.cart_step
            elif a in ("joint+", "joint-"):
                self.joint_target[self.sel_joint] += self.joint_step if a[-1] == "+" else -self.joint_step
            else:  # roll/pitch/yaw ±  (베이스 축 기준 회전)
                axis = _ORI_AXIS[a[:-1]]
                ang = self.ori_step if a[-1] == "+" else -self.ori_step
                self.ori_R = R.from_euler(axis, ang).as_matrix() @ self.ori_R
        return True

    def _apply_control(self):
        arm = self.arm
        if self.mode == "jsc":
            if self.dirty:
                arm.go_joints(self.joint_target); self.dirty = False
        elif self.mode == "tsc":
            if self.dirty:                          # 자세 구속 IK(평행이동 중 자세 유지)
                arm.go_cartesian(self.cart_target, target_R=self.ori_R); self.dirty = False
        elif self.mode == "osc":                    # 매 프레임(월드 높이 보정)
            arm.osc_update(self.cart_target + np.array([0, 0, self.base_z], dtype=np.float32))

    def _status(self):
        self._n += 1
        if self._n % 60 == 0:
            tcp = self.arm.current_tcp()
            extra = f"  [joint_{self.sel_joint+1}]" if self.mode == "jsc" else ""
            print(f"[{self.mode.upper()}] 목표={np.round(self.cart_target,3)} 손끝={np.round(tcp,3)}{extra}")
