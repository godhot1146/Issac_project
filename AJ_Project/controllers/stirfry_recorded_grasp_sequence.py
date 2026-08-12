"""성공한 TSC 기록에서 추출한 볶음 그릇 자동 파지·상승 시퀀스.

``StirfryAutoSequence``의 180 mm 안전 접근과 중력보상 제어는 그대로
재사용한다. 그 뒤에는 고리 안쪽으로 그릇을 한 번에 당기지 않고, 성공
기록처럼 ``조금 회전 -> 그릇 중심 바깥쪽 이동 -> 하강``을 반복한다.
이 계단식 경로는 최종 파지면인 고리 개방부의 옆면을 림에 붙이면서 회전해
한쪽 접촉으로 그릇을 먼저 기울이는 현상을 줄인다. 공용 키보드 텔레오프는
수정하거나 사용하지 않는다.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from stirfry_auto_sequence import StirfryAutoSequence, _PoseStage


class StirfryRecordedGraspSequence(StirfryAutoSequence):
    """기록된 성공 경로로 그릇을 걸고 약 180 mm 들어 올린다."""

    # 성공 TRACE(version 2)의 기준 상태. 실행 시 현재 그릇 위치와의 작은
    # 차이를 모든 Cartesian 목표에 평행 이동해 초기 settling 편차를 흡수한다.
    TRACE_BOWL_REFERENCE_WORLD = np.array(
        [-0.625014, 0.349999, 0.820928], dtype=np.float64
    )

    TRACE_START_Q_XYZW = np.array(
        [0.407863, 0.580954, -0.578299, 0.402132], dtype=np.float64
    )
    TRACE_LOCKED_Q_XYZW = np.array(
        [0.025381, -0.705849, 0.707204, 0.031550], dtype=np.float64
    )

    TRACE_SAFE_START_P_BASE = np.array(
        [-0.414704, 0.364993, 0.405208], dtype=np.float64
    )
    TRACE_RIM_ABOVE_P_BASE = np.array(
        [-0.441704, 0.364993, 0.229210], dtype=np.float64
    )
    TRACE_LOCK_P_BASE = np.array(
        [-0.364705, 0.364993, 0.091210], dtype=np.float64
    )
    TRACE_LOCK_OUT_P_BASE = np.array(
        [-0.358705, 0.364993, 0.091210], dtype=np.float64
    )
    TRACE_LOCK_CENTER_P_BASE = np.array(
        [-0.376705, 0.364993, 0.091210], dtype=np.float64
    )
    TRACE_FIRST_LIFT_P_BASE = np.array(
        [-0.376705, 0.364993, 0.143210], dtype=np.float64
    )
    TRACE_TRANSFER_P_BASE = np.array(
        [-0.335705, 0.364993, 0.143210], dtype=np.float64
    )
    TRACE_SECOND_LIFT_P_BASE = np.array(
        [-0.335705, 0.364993, 0.205210], dtype=np.float64
    )
    TRACE_FINAL_TRANSFER_P_BASE = np.array(
        [-0.314706, 0.364993, 0.205210], dtype=np.float64
    )
    TRACE_FINAL_LIFT_P_BASE = np.array(
        [-0.314706, 0.364993, 0.256210], dtype=np.float64
    )

    # 새 성공 TRACE(39 parts)는 최종 잠금 회전 중 15.99 mm에서 lifted가
    # 처음 true가 되고, 회전 정착 뒤 약 47 mm가 들렸다. 링크6의 정확한
    # 목표 도착보다 이 실제 물리 결과를 우선한다.
    MIN_LOCK_LIFT_M = 0.015
    MIN_FINAL_LIFT_M = 0.150
    FINAL_LIFT_COMMAND_M = 0.051
    MAX_PRELOCK_TILT_DEG = 12.0
    MAX_LOCK_TILT_DEG = 15.0
    TRANSFER_MAX_RELATIVE_POSITION_DRIFT_M = 0.015
    TRANSFER_MAX_RELATIVE_ORIENTATION_DRIFT_DEG = 8.0

    # (이름, 목표 link_6 위치, 목표 자세, 이동시간). 39-part 성공 로그의
    # 미세 키 입력을 물리적으로 의미 있는 끝점만 남겨 압축했다. 각 회전
    # 뒤 +X(그릇 중심 바깥쪽)로 빠지고, 그 상태에서 하강한 다음 다시
    # 회전한다. 마지막 3개 단계는 하부 훅이 잠기기 직전의 미세 정렬이다.
    TRACE_STAIRCASE = (
        ("01 회전", (-0.441704, 0.364993, 0.219210), (0.397708, 0.587884, -0.585329, 0.391932), 1.2),
        ("01 바깥 이동", (-0.430704, 0.364993, 0.219210), (0.397708, 0.587884, -0.585329, 0.391932), 1.0),
        ("01 하강", (-0.430704, 0.364993, 0.208210), (0.397708, 0.587884, -0.585329, 0.391932), 1.4),
        ("02 회전", (-0.430704, 0.364993, 0.208210), (0.384845, 0.596294, -0.593866, 0.379014), 1.2),
        ("02 바깥 이동", (-0.426704, 0.364993, 0.208210), (0.384845, 0.596294, -0.593866, 0.379014), 0.8),
        ("02 하강", (-0.426704, 0.364993, 0.199210), (0.384845, 0.596294, -0.593866, 0.379014), 1.2),
        ("03 회전", (-0.426704, 0.364993, 0.199210), (0.353237, 0.615313, -0.613193, 0.347287), 1.5),
        ("03 바깥 이동", (-0.414704, 0.364993, 0.199210), (0.353237, 0.615313, -0.613193, 0.347287), 1.0),
        ("03 하강", (-0.414704, 0.364993, 0.185210), (0.353237, 0.615313, -0.613193, 0.347287), 1.5),
        ("04 회전", (-0.414704, 0.364993, 0.185210), (0.326153, 0.629876, -0.628017, 0.320117), 1.3),
        ("04 바깥 이동", (-0.406704, 0.364993, 0.185210), (0.326153, 0.629876, -0.628017, 0.320117), 1.0),
        ("04 하강", (-0.406704, 0.364993, 0.169210), (0.326153, 0.629876, -0.628017, 0.320117), 1.6),
        ("05 회전", (-0.406704, 0.364993, 0.169210), (0.292839, 0.645766, -0.644226, 0.286713), 1.4),
        ("05 바깥 이동", (-0.397705, 0.364993, 0.169210), (0.292839, 0.645766, -0.644226, 0.286713), 1.0),
        ("05 하강", (-0.397705, 0.364993, 0.158210), (0.292839, 0.645766, -0.644226, 0.286713), 1.3),
        ("06 회전", (-0.397705, 0.364993, 0.158210), (0.255845, 0.660982, -0.659792, 0.249642), 1.5),
        ("06 바깥 이동", (-0.392705, 0.364993, 0.158210), (0.255845, 0.660982, -0.659792, 0.249642), 0.8),
        ("06 하강", (-0.392705, 0.364993, 0.150210), (0.255845, 0.660982, -0.659792, 0.249642), 1.1),
        ("07 회전", (-0.392705, 0.364993, 0.150210), (0.241390, 0.666271, -0.665216, 0.235163), 0.8),
        ("07 바깥 이동", (-0.387705, 0.364993, 0.150210), (0.241390, 0.666271, -0.665216, 0.235163), 0.8),
        ("07 하강", (-0.387705, 0.364993, 0.142210), (0.241390, 0.666271, -0.665216, 0.235163), 1.1),
        ("08 회전", (-0.387705, 0.364993, 0.142210), (0.218028, 0.674072, -0.673235, 0.211767), 1.0),
        ("08 바깥 이동", (-0.382705, 0.364993, 0.142210), (0.218028, 0.674072, -0.673235, 0.211767), 0.8),
        ("08 하강", (-0.382705, 0.364993, 0.130210), (0.218028, 0.674072, -0.673235, 0.211767), 1.4),
        ("09 회전", (-0.382705, 0.364993, 0.130210), (0.191428, 0.681866, -0.681276, 0.185140), 1.0),
        ("09 바깥 이동", (-0.371705, 0.364993, 0.130210), (0.191428, 0.681866, -0.681276, 0.185140), 1.0),
        ("09 하강", (-0.371705, 0.364993, 0.124210), (0.191428, 0.681866, -0.681276, 0.185140), 1.0),
        ("10 회전", (-0.371705, 0.364993, 0.124210), (0.155511, 0.690621, -0.690361, 0.149200), 1.4),
        ("10 안쪽 미세복귀", (-0.377705, 0.364993, 0.124210), (0.155511, 0.690621, -0.690361, 0.149200), 1.0),
        ("10 하강", (-0.377705, 0.364993, 0.100210), (0.155511, 0.690621, -0.690361, 0.149200), 2.0),
        ("하부 훅 직전 회전", (-0.377705, 0.364993, 0.100210), (-0.119167, -0.697483, 0.697554, -0.112852), 2.0),
        ("하부 훅 바깥 정렬", (-0.369705, 0.364993, 0.100210), (-0.119167, -0.697483, 0.697554, -0.112852), 1.0),
        ("하부 훅 높이 정렬", (-0.369705, 0.364993, 0.091210), (-0.119167, -0.697483, 0.697554, -0.112852), 1.2),
        ("개방부 옆면 밀착", (-0.369705, 0.364993, 0.091210), (-0.094750, -0.700997, 0.701288, -0.088441), 1.2),
        ("최종 잠금 전 바깥 이동", (-0.364705, 0.364993, 0.091210), (-0.094750, -0.700997, 0.701288, -0.088441), 0.8),
    )

    def __init__(
        self,
        gym,
        sim,
        env,
        arm,
        bowl_actor,
        base_z,
        dt=1.0 / 60.0,
        slot_direction_xy=(-1.0, 0.0),
    ):
        self.recorded_initial_bowl_position = None
        self.recorded_initial_bowl_orientation = None
        super().__init__(
            gym,
            sim,
            env,
            arm,
            bowl_actor,
            base_z=base_z,
            dt=dt,
            slot_direction_xy=slot_direction_xy,
            # 기존 --auto와 동일한 안전 접근 웨이포인트를 얻는다. 수동
            # 인계 checkpoint의 처리는 아래 _finish_stage에서 가로챈다.
            manual_handoff=True,
        )
        print(
            "[기록 기반 자동 파지] 안전 접근 뒤 성공 TRACE의 압축 경로로 "
            "그릇 파지와 상승을 실행합니다."
        )
        print(
            "[기록 기반 자동 파지] 39-part 성공 로그의 계단식 경로를 "
            "사용합니다: 소회전 -> 중심 바깥 이동 -> 하강 반복."
        )

    @staticmethod
    def _rotation(quaternion_xyzw):
        quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        return Rotation.from_quat(quaternion).as_matrix()

    def _build_stages(self, bowl_position):
        # 부모가 CAD 기준으로 계산한 첫 180 mm 안전 접근 단계만 재사용한다.
        stages = list(super()._build_stages(bowl_position))
        if len(stages) != 1 or stages[0].checkpoint != "manual_handoff":
            raise RuntimeError("기록 기반 파지의 안전 접근 단계를 만들지 못했습니다.")

        bowl_position = np.asarray(bowl_position, dtype=np.float64)
        self.recorded_initial_bowl_position = bowl_position.copy()
        _, self.recorded_initial_bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        translation = bowl_position - self.TRACE_BOWL_REFERENCE_WORLD
        start_R = self._rotation(self.TRACE_START_Q_XYZW)
        locked_R = self._rotation(self.TRACE_LOCKED_Q_XYZW)

        def trace_stage(
            name,
            duration_s,
            trace_position,
            orientation,
            checkpoint="",
            position_tolerance=0.006,
            orientation_tolerance_deg=1.5,
        ):
            return _PoseStage(
                name=name,
                duration_s=float(duration_s),
                position=(
                    np.asarray(trace_position, dtype=np.float64) + translation
                ).astype(np.float32),
                orientation=np.asarray(orientation, dtype=np.float64),
                checkpoint=checkpoint,
                position_tolerance=position_tolerance,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )

        stages.extend(
            [
                trace_stage(
                    "고공 안전 위치에서 기록 경로 X 정렬",
                    2.0,
                    self.TRACE_SAFE_START_P_BASE,
                    start_R,
                ),
                trace_stage(
                    "그릇 림 위 229mm 링크 목표까지 비접촉 하강",
                    4.0,
                    self.TRACE_RIM_ABOVE_P_BASE,
                    start_R,
                ),
            ]
        )

        for label, position, quaternion, duration_s in self.TRACE_STAIRCASE:
            stages.append(
                trace_stage(
                    f"저기울임 계단 {label}",
                    duration_s,
                    position,
                    self._rotation(quaternion),
                    checkpoint="recorded_stair_step",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                )
            )

        stages.extend(
            [
                trace_stage(
                    "개방부 옆면을 유지하며 하부 훅 최종 잠금",
                    4.0,
                    self.TRACE_LOCK_P_BASE,
                    locked_R,
                    checkpoint="recorded_lock",
                    position_tolerance=0.020,
                    orientation_tolerance_deg=7.0,
                ),
                trace_stage(
                    "잠금 뒤 바깥쪽 6mm 응력 완화",
                    0.8,
                    self.TRACE_LOCK_OUT_P_BASE,
                    locked_R,
                    checkpoint="recorded_replay",
                    position_tolerance=0.020,
                    orientation_tolerance_deg=7.0,
                ),
                trace_stage(
                    "잠금 중심으로 18mm 복귀",
                    1.5,
                    self.TRACE_LOCK_CENTER_P_BASE,
                    locked_R,
                    checkpoint="recorded_replay",
                    position_tolerance=0.020,
                    orientation_tolerance_deg=7.0,
                ),
                trace_stage(
                    "파지 유지 확인용 52mm 상승",
                    2.5,
                    self.TRACE_FIRST_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_replay",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "그릇을 건 상태로 X축 41mm 이동",
                    2.0,
                    self.TRACE_TRANSFER_P_BASE,
                    locked_R,
                    checkpoint="recorded_transfer",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "기록 경로 62mm 중간 상승",
                    2.5,
                    self.TRACE_SECOND_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_replay",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "상승 상태에서 X축 21mm 이동",
                    1.5,
                    self.TRACE_FINAL_TRANSFER_P_BASE,
                    locked_R,
                    checkpoint="recorded_replay",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "기록 경로 51mm 최종 상승",
                    2.5,
                    self.TRACE_FINAL_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_final_lift",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "기록 기반 자동 파지 완료 자세 유지",
                    1.0,
                    self.TRACE_FINAL_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_complete",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
            ]
        )
        return stages

    def _stage_completion_override(
        self, stage, position_error, orientation_error
    ):
        # 이 경로는 키 입력의 시간 순서를 재생한다. 림 접촉 뒤에는 실제
        # link_6가 명령 목표에서 멈추는 것이 정상이라, 각 계단을 목표 자세
        # 도착으로 판정하면 성공 기록에도 10초 timeout이 생긴다.
        if stage.checkpoint in {
            "recorded_stair_step",
            "recorded_replay",
            "recorded_transfer",
        }:
            return "성공 기록의 저속 접촉 구간 재생 완료"

        bowl_lift = self._recorded_bowl_lift()
        if stage.checkpoint == "recorded_lock":
            if bowl_lift < self.MIN_LOCK_LIFT_M:
                return None
            return (
                f"그릇이 {bowl_lift * 1000:.1f} mm 실제 상승해 "
                "파지 성공으로 판정"
            )

        if stage.checkpoint in {"recorded_final_lift", "recorded_complete"}:
            if bowl_lift < self.MIN_FINAL_LIFT_M:
                return None
            return f"그릇 총 상승량 {bowl_lift * 1000:.1f} mm 확인"

        return None

    def _recorded_bowl_lift(self):
        if self.recorded_initial_bowl_position is None:
            return 0.0
        return float(
            self._bowl_position()[2]
            - self.recorded_initial_bowl_position[2]
        )

    def _recorded_bowl_tilt_deg(self):
        if self.recorded_initial_bowl_orientation is None:
            return 0.0
        _, bowl_orientation = self._rigid_body_pose(self.bowl_actor, 0)
        initial_up = self.recorded_initial_bowl_orientation[:, 2]
        current_up = bowl_orientation[:, 2]
        cosine = float(np.clip(np.dot(initial_up, current_up), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _finish_stage(self, stage):
        if stage.checkpoint == "manual_handoff":
            # 기존 --auto는 여기서 키보드로 넘기지만 --auto-grasp는 같은
            # 중력보상 제어기를 유지한 채 기록 경로의 다음 단계로 이어간다.
            self.arm._last_q = stage.joint_target.copy()
            self.handoff_ready = False
            print(
                "[기록 기반 자동 파지] 180mm 안전 접근 완료. "
                "모드 전환 없이 자동 파지 경로를 시작합니다."
            )
            self._enter_stage(self.stage_index + 1)
            return

        if stage.checkpoint == "recorded_stair_step":
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            if bowl_tilt_deg > self.MAX_PRELOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "계단식 접근 중 그릇 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커져 파지를 중단합니다."
                )
                return

        if stage.checkpoint == "recorded_lock":
            bowl_lift = self._recorded_bowl_lift()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            print(
                "[기록 기반 자동 파지][잠금 검사] "
                f"그릇 상승량 {bowl_lift * 1000:.1f} mm, "
                f"초기 대비 기울기 {bowl_tilt_deg:.2f} deg"
            )
            if bowl_lift < self.MIN_LOCK_LIFT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    f"피치 잠금 뒤 그릇이 {self.MIN_LOCK_LIFT_M * 1000:.0f}mm "
                    "이상 상승하지 않아 "
                    "수평 이동을 시작하지 않습니다."
                )
                return
            if bowl_tilt_deg > self.MAX_LOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "그릇은 들렸지만 잠금 직후 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커서 안정 파지로 판정하지 "
                    "않습니다."
                )
                return

            # 물리 접촉 때문에 기록 목표와 다른 자세에서 멈춘 경우 그
            # 실제 성공 자세를 다음 안정화·수평 이동·상승 단계의 원점으로
            # 사용한다. 이후에 다시 기록 목표로 당기면서 파지를 놓치는
            # 현상을 막는다.
            translation_mm, orientation_delta_deg = (
                self._rebase_future_from_actual_stage(stage)
            )
            print(
                "[기록 기반 자동 파지][잠금 재정렬] 실제 파지 자세 기준: "
                f"위치 {translation_mm:.1f} mm, "
                f"자세 {orientation_delta_deg:.2f} deg 보정"
            )

        if stage.checkpoint == "recorded_transfer":
            if not self._validate_transfer_retention(stage):
                return

        if stage.checkpoint == "recorded_final_lift":
            if not self._validate_lift_follow(
                stage,
                "기록 경로 51mm 최종 상승",
                self.FINAL_LIFT_COMMAND_M,
            ):
                return

        if stage.checkpoint == "recorded_complete":
            bowl_position = self._bowl_position()
            bowl_lift = float(
                bowl_position[2] - self.recorded_initial_bowl_position[2]
            )
            if bowl_lift < self.MIN_FINAL_LIFT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "완료 시점의 그릇 총 상승량이 150mm보다 작아 "
                    "자동 파지 성공으로 판정하지 않습니다."
                )
                return
            self.finished = True
            print(
                "[기록 기반 자동 파지][완료] 그릇을 안정적으로 들어 "
                f"올렸습니다. 최종 상승량={bowl_lift * 1000:.1f} mm, "
                f"초기 대비 기울기={self._recorded_bowl_tilt_deg():.2f} deg"
            )
            return

        super()._finish_stage(stage)

    def _print_pose_diagnostics(self, stage):
        super()._print_pose_diagnostics(stage)
        if self.recorded_initial_bowl_position is not None:
            print(
                "  그릇 총 상승량 = "
                f"{self._recorded_bowl_lift() * 1000:.1f} mm"
            )
            print(
                "  그릇 초기 대비 기울기 = "
                f"{self._recorded_bowl_tilt_deg():.2f} deg"
            )

    def _validate_transfer_retention(self, stage):
        """수평 이동 중 bowl/gripper 상대 자세가 유지됐는지 검사한다."""
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        bowl_in_gripper = gripper_orientation.T @ (
            bowl_position - gripper_position
        )
        relative_position_drift = float(
            np.linalg.norm(bowl_in_gripper - self.stage_start_bowl_in_gripper)
        )
        bowl_orientation_in_gripper = gripper_orientation.T @ bowl_orientation
        relative_orientation_drift = float(
            np.degrees(
                Rotation.from_matrix(
                    bowl_orientation_in_gripper
                    @ self.stage_start_bowl_orientation_in_gripper.T
                ).magnitude()
            )
        )
        print(
            "[기록 기반 자동 파지][수평 이동 추종 검사] "
            f"상대 위치 변화 {relative_position_drift * 1000:.1f} mm, "
            f"상대 자세 변화 {relative_orientation_drift:.2f} deg"
        )
        if (
            relative_position_drift
            > self.TRANSFER_MAX_RELATIVE_POSITION_DRIFT_M
            or relative_orientation_drift
            > self.TRANSFER_MAX_RELATIVE_ORIENTATION_DRIFT_DEG
        ):
            self._print_grasp_diagnostics(stage)
            self._fail(
                "수평 이동 중 그릇과 그리퍼의 상대 자세가 크게 변해 "
                "파지 이탈로 판정합니다."
            )
            return False
        return True
