"""성공한 TSC 기록에서 추출한 볶음 그릇 자동 파지·상승 시퀀스.

``StirfryAutoSequence``의 180 mm 안전 접근과 중력보상 제어는 그대로
재사용한다. 그 뒤에는 수동 기록의 모든 키 입력을 재생하지 않고, 실제로
파지에 기여한 링크6 목표 자세만 연속 보간한다. 공용 키보드 텔레오프는
수정하거나 사용하지 않는다.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from stirfry_auto_sequence import StirfryAutoSequence, _PoseStage


class StirfryRecordedGraspSequence(StirfryAutoSequence):
    """기록된 성공 경로로 그릇을 걸고 약 190 mm 들어 올린다."""

    # 성공 TRACE(version 2)의 기준 상태. 실행 시 현재 그릇 위치와의 작은
    # 차이를 모든 Cartesian 목표에 평행 이동해 초기 settling 편차를 흡수한다.
    TRACE_BOWL_REFERENCE_WORLD = np.array(
        [-0.625014, 0.349999, 0.820928], dtype=np.float64
    )

    TRACE_START_Q_XYZW = np.array(
        [0.407863, 0.580954, -0.578299, 0.402132], dtype=np.float64
    )
    TRACE_EARLY_LOCK_Q_XYZW = np.array(
        [0.361233, 0.610714, -0.608517, 0.355312], dtype=np.float64
    )
    TRACE_LOCKED_Q_XYZW = np.array(
        [0.031552, -0.705547, 0.706956, 0.037708], dtype=np.float64
    )

    TRACE_SAFE_START_P_BASE = np.array(
        [-0.414704, 0.364993, 0.405208], dtype=np.float64
    )
    TRACE_RIM_ABOVE_P_BASE = np.array(
        [-0.441704, 0.364993, 0.229210], dtype=np.float64
    )
    TRACE_RIM_ENTRY_P_BASE = np.array(
        [-0.441704, 0.364993, 0.203210], dtype=np.float64
    )
    TRACE_LOCK_P_BASE = np.array(
        [-0.441704, 0.364993, 0.178210], dtype=np.float64
    )
    TRACE_TRANSFER_P_BASE = np.array(
        [-0.372705, 0.364993, 0.178210], dtype=np.float64
    )
    TRACE_LOWER_CORRECTION_P_BASE = np.array(
        [-0.372705, 0.364993, 0.157210], dtype=np.float64
    )
    TRACE_FINAL_LIFT_P_BASE = np.array(
        [-0.372705, 0.364993, 0.227210], dtype=np.float64
    )

    MIN_LOCK_LIFT_M = 0.030
    MIN_FINAL_LIFT_M = 0.150
    FINAL_LIFT_COMMAND_M = 0.070
    TRANSFER_MAX_RELATIVE_POSITION_DRIFT_M = 0.015
    TRANSFER_MAX_RELATIVE_ORIENTATION_DRIFT_DEG = 8.0

    # 고리 끝을 완전히 고정한 채 66.5도를 회전하면 손목이 약 160 mm
    # 아래로 원호 이동해 테이블과 충돌할 수 있다. 성공 기록의 궤적은
    # 유지하되, 한쪽 고리만 걸려 있는 회전 중간에만 접촉점 중심 원호를
    # 일부 혼합한다. 시작/끝에서는 보상이 0이므로 기록된 최종 파지
    # 자세와 뒤 웨이포인트는 바뀌지 않는다.
    LOCK_PIVOT_COMPENSATION = 0.35

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
        self.recorded_lock_pivot_world = None
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
            "[기록 기반 자동 파지] 잠금 회전 중 고리 접촉점 원호 보상 "
            f"최대 {self.LOCK_PIVOT_COMPENSATION * 100:.0f}%를 적용해 "
            "그릇 기울어짐을 줄입니다."
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
        translation = bowl_position - self.TRACE_BOWL_REFERENCE_WORLD
        start_R = self._rotation(self.TRACE_START_Q_XYZW)
        early_lock_R = self._rotation(self.TRACE_EARLY_LOCK_Q_XYZW)
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
                trace_stage(
                    "기록된 림 진입 높이까지 저속 하강",
                    2.5,
                    self.TRACE_RIM_ENTRY_P_BASE,
                    start_R,
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                trace_stage(
                    "림을 따라 25mm 하강하며 초기 피치 걸기",
                    4.0,
                    self.TRACE_LOCK_P_BASE,
                    early_lock_R,
                    position_tolerance=0.012,
                    orientation_tolerance_deg=4.0,
                ),
                trace_stage(
                    "위치를 유지하며 기록된 피치 잠금 자세까지 회전",
                    9.0,
                    self.TRACE_LOCK_P_BASE,
                    locked_R,
                    checkpoint="recorded_lock",
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "잠금 접촉 안정화",
                    1.0,
                    self.TRACE_LOCK_P_BASE,
                    locked_R,
                    position_tolerance=0.015,
                    orientation_tolerance_deg=5.0,
                ),
                trace_stage(
                    "그릇을 건 상태로 X축 69mm 이동",
                    3.0,
                    self.TRACE_TRANSFER_P_BASE,
                    locked_R,
                    checkpoint="recorded_transfer",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                trace_stage(
                    "기록된 상승 전 21mm 하강 보정",
                    1.5,
                    self.TRACE_LOWER_CORRECTION_P_BASE,
                    locked_R,
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                trace_stage(
                    "그릇 최종 70mm 상승",
                    3.0,
                    self.TRACE_FINAL_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_final_lift",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                trace_stage(
                    "기록 기반 자동 파지 완료 자세 유지",
                    1.0,
                    self.TRACE_FINAL_LIFT_P_BASE,
                    locked_R,
                    checkpoint="recorded_complete",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
            ]
        )
        return stages

    def _enter_stage(self, index):
        super()._enter_stage(index)
        stage = self.stages[index]
        if stage.checkpoint != "recorded_lock":
            return

        # 계획 좌표가 아니라 실제 초기 접촉 자세에서 고리 피벗을 잡는다.
        # 앞 단계의 수 mm 정착 오차가 원호 보상에 누적되는 것을 막는다.
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        self.recorded_lock_pivot_world = (
            gripper_position
            + gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        print(
            "[기록 기반 자동 파지][기울어짐 억제] 실제 고리 접촉점을 "
            "기준으로 잠금 원호 보상을 시작합니다."
        )

    def _command_stage(self, stage, alpha):
        if (
            stage.checkpoint != "recorded_lock"
            or self.recorded_lock_pivot_world is None
        ):
            super()._command_stage(stage, alpha)
            return

        # 기록 경로의 고정 link_6 위치와, 고리 접촉점을 완전히 고정하는
        # 원호 위치를 혼합한다. 4a(1-a) 창을 사용해 회전 중간에만 보상하고
        # 양쪽 고리가 잠기는 마지막에는 원래 성공 경로로 자연스럽게
        # 돌아온다.
        target_orientation = self._slerp_matrix(
            self.stage_start_orientation, stage.orientation, alpha
        )
        recorded_position = self._lerp(
            self.stage_start_position, stage.position, alpha
        )
        gripper_orientation = (
            target_orientation @ self.LINK6_TO_GRIPPER_ROTATION
        )
        pivot_gripper_position = (
            self.recorded_lock_pivot_world
            - gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        pivot_position, _ = self._link6_pose(
            pivot_gripper_position, gripper_orientation
        )
        window = 4.0 * alpha * (1.0 - alpha)
        blend = self.LOCK_PIVOT_COMPENSATION * window
        target_position = self._lerp(
            recorded_position, pivot_position, blend
        )

        self.command_position = np.asarray(target_position, dtype=np.float64)
        self.command_orientation = np.asarray(
            target_orientation, dtype=np.float64
        )
        target_joints, _, ik_error_mm = self.arm.solve_ik(
            target_position,
            target_R=target_orientation,
            seed_6=self.arm._last_q,
        )
        if not np.all(np.isfinite(target_joints)) or ik_error_mm > 8.0:
            self._fail(
                f"'{stage.name}' 고리 피벗 보상 중 IK 실패: "
                f"위치 오차 {ik_error_mm:.2f} mm"
            )
            return
        self.arm._last_q = target_joints.copy()
        settling = self.stage_elapsed + 1.0e-9 >= stage.duration_s
        self.auto_control.command(target_joints, settling=settling)

    def _stage_completion_override(
        self, stage, position_error, orientation_error
    ):
        if stage.checkpoint != "recorded_lock":
            return None
        bowl_lift = self._recorded_bowl_lift()
        if bowl_lift < self.MIN_LOCK_LIFT_M:
            return None
        return (
            f"그릇이 {bowl_lift * 1000:.1f} mm 실제 상승해 "
            "파지 성공으로 판정"
        )

    def _recorded_bowl_lift(self):
        if self.recorded_initial_bowl_position is None:
            return 0.0
        return float(
            self._bowl_position()[2]
            - self.recorded_initial_bowl_position[2]
        )

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

        if stage.checkpoint == "recorded_lock":
            bowl_lift = self._recorded_bowl_lift()
            print(
                "[기록 기반 자동 파지][잠금 검사] "
                f"그릇 상승량 {bowl_lift * 1000:.1f} mm"
            )
            if bowl_lift < self.MIN_LOCK_LIFT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "피치 잠금 뒤 그릇이 30mm 이상 상승하지 않아 "
                    "수평 이동을 시작하지 않습니다."
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
                "기록 경로 70mm 최종 상승",
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
                f"올렸습니다. 최종 상승량={bowl_lift * 1000:.1f} mm"
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
