"""두산 A0509 볶음 그릇의 접촉 기반 자동 파지·붓기 시퀀스.

이 그리퍼는 열고 닫히는 집게가 아니다. 세운 그리퍼의 두꺼운 상부 고리에
그릇 림을 먼저 끼운 뒤, 그 접점을 유지하며 손목축을 회전해 아래쪽 훅까지
걸리게 하는 고정형 그리퍼다. 따라서 이 모듈은 bowl actor를 순간이동시키거나
로봇에 가짜로 고정하지 않고, CAD에서 검증한 상·하 접촉 형상과 PhysX 접촉만
사용한다.

대상은 그리퍼가 맞도록 설계된 원본 크기(Ø250 mm)의 조리 그릇이다.
준비 테이블의 재료 그릇들은 0.75배로 축소되어 있어 같은 파지 좌표를
사용할 수 없다.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

from stirfry_auto_control import StirfryAutoJointControl


@dataclass(frozen=True)
class _PoseStage:
    name: str
    duration_s: float
    position: np.ndarray
    orientation: np.ndarray
    joint_target: object = None
    checkpoint: str = ""
    pivot_world: object = None
    pivot_end_world: object = None
    pivot_local: object = None
    pivot_start_deg: object = None
    pivot_end_deg: object = None
    pivot_motion_start: float = 0.0
    contact_seek: bool = False
    position_tolerance: object = None
    orientation_tolerance_deg: object = None


class StirfryAutoSequence:
    """그릇 하나를 실제 접촉으로 걸어 들어 올린 뒤 한 번 기울인다."""

    HOME_Q = np.zeros(6, dtype=np.float32)

    # 최대 접촉 탐색 하강점에서 계산한 잠금 자세의 bowl origin(m).
    # 실제 실행에서는 천장 접촉을 감지한 위치만큼 뒤 웨이포인트를 재정렬한다.
    BOWL_FROM_GRIPPER_LOCKED = np.array(
        [0.253382480, 0.0, -0.001173198], dtype=np.float64
    )

    # 림보다 10 mm 위에서 75도까지만 살짝 연다. 이후 월드 수평 거리만
    # 명령하지 않고, STEP 고리 입구의 아래 4.00 mm / 위 31.469 mm 사이
    # 중앙점(x=148.0, z=17.7345 mm)에 림 기준점을 직접 정렬한다. 이
    # 중앙에서 먼저 7.326 mm 하강하고 3도 안쪽으로 되돌린다. 그다음 최대
    # 50 mm를 천천히 추가 하강하되 추종 오차로 천장 접촉을 감지하면 즉시
    # 멈추고, 실제 접촉 위치를 기준으로 잠금·상승 웨이포인트를 재정렬한다.
    VERTICAL_ENTRY_DEG = 70.0
    UPPER_OPEN_DEG = 75.0
    UPPER_INWARD_DEG = 72.0
    LOCK_DEG = -2.32

    VERTICAL_STAGING_CLEARANCE = 0.18
    UPPER_PRECONTACT_CLEARANCE = 0.04
    UPPER_INSERT_CLEARANCE = 0.010
    UPPER_CENTER_DESCENT = 0.007326201
    UPPER_CONTACT_MAX_DESCENT = 0.050
    DIAGONAL_SEAT_CYCLES = 3
    DIAGONAL_CENTER_ADVANCE = 0.005
    DIAGONAL_DESCENT = 0.010
    DIAGONAL_POSITION_TOLERANCE = 0.006
    DIAGONAL_CONTACT_ERROR = 0.007
    DIAGONAL_CONTACT_ORIENTATION_DEG = 2.0
    FINAL_LOCK_ROTATION_DEG = 2.5
    TEST_LIFT_HEIGHT = 0.04
    INTERMEDIATE_LIFT_HEIGHT = 0.15
    LIFT_HEIGHT = 0.45
    POUR_DEG = 50.0

    # all-zero home에서 바로 Cartesian IK를 시작하면 팔꿈치가 스탠드 안쪽
    # 아래로 접히는 해를 고른다. 첫 staging pose는 이 seed로 elbow-up 해를
    # 고른 뒤 관절공간으로 이동하고, 이후 IK가 그 분기를 이어받게 한다.
    ELBOW_UP_SEED = np.deg2rad(
        np.array([-63.0, -40.0, -90.0, 30.0, -120.0, -66.0], dtype=np.float64)
    ).astype(np.float32)

    POSITION_TOLERANCE = 0.005
    ORIENTATION_TOLERANCE_DEG = 1.0
    ARRIVAL_TIMEOUT_S = 10.0
    CONTACT_DETECT_ERROR = 0.006
    CONTACT_DETECT_HOLD_S = 0.6
    CONTACT_DETECT_ORIENTATION_DEG = 3.0
    LIFT_MIN_COMMAND_FRACTION = 0.70
    LIFT_MIN_FOLLOW_RATIO = 0.70
    LIFT_MAX_RELATIVE_POSITION_DRIFT = 0.008
    LIFT_MAX_RELATIVE_ORIENTATION_DRIFT_DEG = 6.0

    # 통합 URDF의 link_6 -> gripper fixed joint.
    LINK6_TO_GRIPPER_POSITION = np.array([0.0, 0.0, 0.016], dtype=np.float64)
    LINK6_TO_GRIPPER_ROTATION = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()

    # CAD 좌표(m): 최초 림 정렬 기준, 27.47 mm 입구의 중앙점, 윗고리
    # 접촉 피벗축이다. 입구 중앙은 x=148 mm 단면에서 아래 z=4.000 mm와
    # 위 z=31.469 mm의 중간값을 사용한다.
    UPPER_ENTRY_REFERENCE_LOCAL = np.array(
        [0.1545, 0.0, 0.0355], dtype=np.float64
    )
    UPPER_MOUTH_CENTER_LOCAL = np.array(
        [0.148, 0.0, 0.0177345], dtype=np.float64
    )
    UPPER_HOOK_SEAT_LOCAL = np.array(
        [0.150, 0.0, 0.031], dtype=np.float64
    )
    BOWL_NEAR_RIM_LOCAL = np.array([-0.125, 0.0, 0.080], dtype=np.float64)

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
        manual_handoff=False,
    ):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.arm = arm
        self.bowl_actor = bowl_actor
        self.base_z = float(base_z)
        self.dt = float(dt)
        self.manual_handoff = bool(manual_handoff)

        arm_body_names = self.gym.get_actor_rigid_body_names(self.env, self.arm.actor)
        self.link6_body_index = arm_body_names.index("link_6")
        self.gripper_body_index = arm_body_names.index("stirfry_gripper_link")

        # gripper +X가 슬롯을 따라 그릇 중심을 향한다. +Z는 world up이고
        # +Y는 두 벡터로부터 만든 손목 회전축이다.
        x_axis = np.array(
            [slot_direction_xy[0], slot_direction_xy[1], 0.0], dtype=np.float64
        )
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        self.bowl_frame = np.column_stack((x_axis, y_axis, z_axis))

        self.home_frames = max(1, int(round(2.0 / self.dt)))
        self.home_frame = 0
        self.stages = []
        self.stage_index = -1
        self.stage_elapsed = 0.0
        self.arrival_wait = 0.0
        self.wait_report_second = 0
        self.stage_start_position = None
        self.stage_start_orientation = None
        self.stage_start_joints = None
        self.stage_start_bowl_position = None
        self.stage_start_gripper_position = None
        self.stage_start_bowl_in_gripper = None
        self.stage_start_bowl_orientation_in_gripper = None
        self.finished = False
        self.failed = False
        self.handoff_ready = False
        self.expected_seated_bowl_in_gripper = None
        self.command_position = None
        self.command_orientation = None
        self.contact_stall_time = 0.0
        self.upper_contact_detected = False
        self.lower_contact_detected = False

        self.arm.go_joints(self.HOME_Q)
        self.auto_control = StirfryAutoJointControl(self.arm, dt=self.dt)
        self.auto_control.command(self.HOME_Q)
        if self.manual_handoff:
            print(
                "[자동 접근] 그리퍼를 조리 그릇 앞 비접촉 위치까지 이동한 뒤 "
                "키보드 미세조작으로 전환합니다."
            )
        else:
            print(
                "[자동] 원본 크기 조리 그릇 1개를 대상으로 "
                "접촉 기반 파지·붓기를 시작합니다."
            )

    def update(self):
        """시뮬레이션 루프에서 물리 스텝 전에 매 프레임 한 번 호출한다."""
        if self.finished:
            self.auto_control.update()
            return

        if self.stage_index < 0:
            self._update_home()
            return

        stage = self.stages[self.stage_index]
        alpha = min(1.0, self.stage_elapsed / stage.duration_s)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
        self._command_stage(stage, alpha)
        if self.finished:
            return

        if stage.contact_seek and self.stage_elapsed >= 1.0:
            command_position_error, command_orientation_error = (
                self._command_pose_error()
            )
            orientation_limit = (
                8.0
                if stage.checkpoint == "lower_hook_contact_seek"
                else self.CONTACT_DETECT_ORIENTATION_DEG
            )
            if (
                command_position_error >= self.CONTACT_DETECT_ERROR
                and command_orientation_error <= orientation_limit
            ):
                self.contact_stall_time += self.dt
            else:
                self.contact_stall_time = 0.0
            if self.contact_stall_time >= self.CONTACT_DETECT_HOLD_S:
                if stage.checkpoint == "lower_hook_contact_seek":
                    print(
                        "[자동][하부 훅 접촉 감지] "
                        f"위치 오차 {command_position_error * 1000:.1f} mm, "
                        f"자세 오차 {command_orientation_error:.2f} deg가 "
                        f"{self.contact_stall_time:.1f}s 지속되어 잠금 회전을 멈춥니다."
                    )
                    self.lower_contact_detected = True
                    self._rebase_after_lower_contact()
                else:
                    print(
                        "[자동][천장 접촉 감지] "
                        f"추종 오차 {command_position_error * 1000:.1f} mm가 "
                        f"{self.contact_stall_time:.1f}s 지속되어 추가 하강을 멈춥니다."
                    )
                    self.upper_contact_detected = True
                    self._rebase_after_upper_contact()
                self._finish_stage(stage)
                return

        if self.stage_elapsed + 1.0e-9 < stage.duration_s:
            self.stage_elapsed += self.dt
            return

        # 예정 시간이 지난 뒤에는 실제 rigid body가 목표에 도착할 때까지
        # 최종 명령을 유지한다. 로그의 단계 번호만 진행하는 상황을 막는다.
        position_error, orientation_error = self._stage_error(stage)
        position_tolerance = (
            self.POSITION_TOLERANCE
            if stage.position_tolerance is None
            else float(stage.position_tolerance)
        )
        orientation_tolerance = (
            self.ORIENTATION_TOLERANCE_DEG
            if stage.orientation_tolerance_deg is None
            else float(stage.orientation_tolerance_deg)
        )
        completion_override = self._stage_completion_override(
            stage, position_error, orientation_error
        )
        if completion_override is not None:
            print(
                f"[자동][조건 충족] {stage.name}: {completion_override} "
                f"(실제 자세 오차: 위치 {position_error * 1000:.1f} mm, "
                f"자세 {orientation_error:.2f} deg)"
            )
            self._finish_stage(stage)
            return
        if (
            position_error <= position_tolerance
            and orientation_error <= orientation_tolerance
        ):
            print(
                f"[자동][도착] {stage.name}: 위치 {position_error * 1000:.1f} mm, "
                f"자세 {orientation_error:.2f} deg"
            )
            self._finish_stage(stage)
            return

        self.arrival_wait += self.dt
        report_second = int(self.arrival_wait)
        if report_second > self.wait_report_second:
            self.wait_report_second = report_second
            print(
                f"[자동][도착 대기 {report_second:02d}s] {stage.name}: "
                f"위치 {position_error * 1000:.1f} mm, 자세 {orientation_error:.2f} deg"
            )
        diagonal_contact = (
            stage.checkpoint == "diagonal_seat"
            and self.arrival_wait >= 0.8
            and position_error >= self.DIAGONAL_CONTACT_ERROR
            and orientation_error <= self.DIAGONAL_CONTACT_ORIENTATION_DEG
        )
        final_lock_contact = (
            stage.checkpoint == "final_lock_rotation"
            and self.arrival_wait >= 0.8
        )
        if diagonal_contact or final_lock_contact:
            print(
                "[자동][하부 훅 안착 저항] "
                f"{stage.name}: 위치 오차 {position_error * 1000:.1f} mm, "
                f"자세 오차 {orientation_error:.2f} deg. "
                "실제 정지 자세를 접촉 위치로 사용합니다."
            )
            if stage.checkpoint == "diagonal_seat":
                self._retarget_final_lock_from_current()
                return
            self._rebase_future_from_actual_stage(stage)
            self._finish_stage(stage)
            return
        if self.arrival_wait >= self.ARRIVAL_TIMEOUT_S:
            self._print_pose_diagnostics(stage)
            self._fail(
                f"'{stage.name}' 실제 목표 미도착: 위치 {position_error * 1000:.1f} mm, "
                f"자세 {orientation_error:.2f} deg"
            )

    def _stage_completion_override(
        self, stage, position_error, orientation_error
    ):
        """물리적 작업 결과로 자세 도착 조건을 대체할 확장 지점.

        기본 자동 시퀀스는 항상 실제 link_6 자세 도착을 요구한다. 접촉으로
        목표 자세에 도달할 수 없어도 작업 결과를 직접 검증할 수 있는 파생
        시퀀스만 이 메서드를 재정의한다.
        """
        return None

    def _command_stage(self, stage, alpha):
        settling = self.stage_elapsed + 1.0e-9 >= stage.duration_s
        if stage.joint_target is not None:
            target_joints = self._lerp(
                self.stage_start_joints, stage.joint_target, alpha
            )
            self.auto_control.command(target_joints, settling=settling)
            return

        if stage.pivot_world is not None:
            target_position, target_orientation = self._pivot_link6_pose(
                stage, alpha
            )
        else:
            target_position = self._lerp(
                self.stage_start_position, stage.position, alpha
            )
            target_orientation = self._slerp_matrix(
                self.stage_start_orientation, stage.orientation, alpha
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
                f"'{stage.name}' 이동 중 IK 실패: 위치 오차 {ik_error_mm:.2f} mm"
            )
            return
        self.arm._last_q = target_joints.copy()
        self.auto_control.command(target_joints, settling=settling)

    def _update_home(self):
        self.auto_control.command(self.HOME_Q)
        self.home_frame += 1
        if self.home_frame < self.home_frames:
            return

        bowl_position = self._bowl_position()
        self.stages = self._build_stages(bowl_position)
        self._validate_goals()
        print(
            "[자동] 그릇 기준점 = "
            f"{np.round(bowl_position, 4)} m, 상부부터 접촉하도록 진입합니다."
        )
        self._enter_stage(0)

    def _build_stages(self, bowl_position):
        # 림보다 10 mm 위에서 75도로 고리 입구를 살짝 연다. 그다음 림
        # 기준점을 27.47 mm 입구의 중앙에 직접 맞추고 7.3 mm 하강한다.
        # 이어 3도 안쪽으로 되돌린 뒤 최대 50 mm를 천천히 하강한다.
        # 추종 오차가 천장 접촉을 나타내면 실제 정지 위치를 기준으로 뒤의
        # 잠금·상승 경로를 재정렬한다.
        outer_rim_world = (
            bowl_position + self.bowl_frame @ self.BOWL_NEAR_RIM_LOCAL
        )
        entry_R = self._gripper_rotation(self.VERTICAL_ENTRY_DEG)
        open_R = self._gripper_rotation(self.UPPER_OPEN_DEG)
        inward_R = self._gripper_rotation(self.UPPER_INWARD_DEG)
        lock_R = self._gripper_rotation(self.LOCK_DEG)
        entry_origin = (
            outer_rim_world
            - entry_R @ self.UPPER_ENTRY_REFERENCE_LOCAL
        )
        open_origin = (
            entry_origin
            + self.bowl_frame[:, 2] * self.UPPER_INSERT_CLEARANCE
        )
        centered_origin = (
            outer_rim_world
            - open_R @ self.UPPER_MOUTH_CENTER_LOCAL
        )
        seated_origin = (
            centered_origin
            - self.bowl_frame[:, 2] * self.UPPER_CENTER_DESCENT
        )
        deep_origin = (
            seated_origin
            - self.bowl_frame[:, 2] * self.UPPER_CONTACT_MAX_DESCENT
        )
        upper_seat_pivot_world = (
            deep_origin + inward_R @ self.UPPER_HOOK_SEAT_LOCAL
        )
        locked_origin = (
            upper_seat_pivot_world
            - lock_R @ self.UPPER_HOOK_SEAT_LOCAL
        )
        locked_bowl_in_gripper = lock_R.T @ (bowl_position - locked_origin)
        locked_geometry_error = np.linalg.norm(
            locked_bowl_in_gripper - self.BOWL_FROM_GRIPPER_LOCKED
        )
        if locked_geometry_error > 0.0002:
            raise RuntimeError(
                "상부 천장 피벗의 최종 잠금 자세가 CAD 기준과 일치하지 않습니다: "
                f"{locked_geometry_error * 1000:.2f} mm"
            )
        staging_origin = entry_origin + np.array(
            [0.0, 0.0, self.VERTICAL_STAGING_CLEARANCE]
        )
        precontact_origin = entry_origin + np.array(
            [0.0, 0.0, self.UPPER_PRECONTACT_CLEARANCE]
        )
        def stage(
            name,
            duration_s,
            gripper_position,
            gripper_orientation,
            joint_target=None,
            checkpoint="",
            pivot_world=None,
            pivot_local=None,
            pivot_start_deg=None,
            pivot_end_deg=None,
            contact_seek=False,
            position_tolerance=None,
            orientation_tolerance_deg=None,
        ):
            link_position, link_orientation = self._link6_pose(
                gripper_position, gripper_orientation
            )
            return _PoseStage(
                name,
                duration_s,
                link_position.astype(np.float32),
                link_orientation.astype(np.float64),
                joint_target=joint_target,
                checkpoint=checkpoint,
                pivot_world=pivot_world,
                pivot_local=pivot_local,
                pivot_start_deg=pivot_start_deg,
                pivot_end_deg=pivot_end_deg,
                contact_seek=contact_seek,
                position_tolerance=position_tolerance,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )

        staging_position, staging_orientation = self._link6_pose(
            staging_origin, entry_R
        )
        staging_q, _, staging_error = self.arm.solve_ik(
            staging_position,
            target_R=staging_orientation,
            seed_6=self.ELBOW_UP_SEED,
        )
        if staging_error > 8.0:
            raise RuntimeError(
                f"elbow-up staging IK 실패: 위치 오차 {staging_error:.2f} mm"
            )

        stages = [
            stage(
                (
                    "그릇 림 접촉 기준보다 180mm 위 수동 인계 대기"
                    if self.manual_handoff
                    else "그릇 위 수직 파지 대기(elbow-up)"
                ),
                4.0,
                staging_origin,
                entry_R,
                joint_target=staging_q,
                checkpoint=("manual_handoff" if self.manual_handoff else ""),
            ),
            stage(
                "두꺼운 상부 고리 위로 수직 하강",
                3.0,
                precontact_origin,
                entry_R,
            ),
            stage(
                "두꺼운 상부 고리를 림보다 10mm 위로 하강",
                2.0,
                open_origin,
                entry_R,
            ),
            stage(
                "잠금 반대 회전으로 상부 고리 입구 열기",
                2.0,
                open_origin,
                open_R,
            ),
            stage(
                "림을 27.5mm 고리 입구의 정중앙으로 삽입",
                4.0,
                centered_origin,
                open_R,
            ),
            stage(
                "입구 중앙에서 윗고리 접촉 직전까지 7.3mm 하강",
                3.0,
                seated_origin,
                open_R,
            ),
            stage(
                "고리를 안쪽으로 3도 되돌리기",
                2.0,
                seated_origin,
                inward_R,
            ),
            stage(
                "천장 접촉까지 최대 50mm 추가 수직 하강",
                8.0,
                deep_origin,
                inward_R,
                checkpoint="upper_contact_seek",
                contact_seek=True,
            ),
            stage(
                "상부 고리 천장 접촉 안정화",
                2.0,
                deep_origin,
                inward_R,
            ),
            stage(
                "상부 접점을 유지하며 하부 훅 회전 잠금",
                8.0,
                locked_origin,
                lock_R,
                pivot_world=upper_seat_pivot_world,
                pivot_local=self.UPPER_HOOK_SEAT_LOCAL,
                pivot_start_deg=self.UPPER_INWARD_DEG,
                pivot_end_deg=self.LOCK_DEG,
                checkpoint="lower_hook_contact_seek",
                contact_seek=True,
            ),
        ]

        # 큰 잠금 회전이 림 옆면에서 멈추면 회전을 더 반복하지 않는다.
        # 손목 자세를 고정하고 그리퍼 전체를 그릇 중심 방향 5 mm, 아래
        # 10 mm의 대각선으로 최대 세 번 이동해 상부 고리 안쪽으로 림을
        # 밀어 넣는다. 이동 저항이 감지되면 남은 이동을 건너뛰고 그 실제
        # 접촉점에서 마지막 2.5도 잠금 회전만 수행한다.
        diagonal_step = (
            self.bowl_frame[:, 0] * self.DIAGONAL_CENTER_ADVANCE
            - self.bowl_frame[:, 2] * self.DIAGONAL_DESCENT
        )
        seated_origin = locked_origin.copy()
        seated_R = lock_R.copy()
        for cycle in range(1, self.DIAGONAL_SEAT_CYCLES + 1):
            seated_origin = seated_origin + diagonal_step
            stages.append(
                stage(
                    f"대각선 안착 {cycle}/{self.DIAGONAL_SEAT_CYCLES}: "
                    f"중심 {self.DIAGONAL_CENTER_ADVANCE * 1000:.0f}mm + "
                    f"하강 {self.DIAGONAL_DESCENT * 1000:.0f}mm",
                    3.0,
                    seated_origin,
                    seated_R,
                    checkpoint="diagonal_seat",
                    position_tolerance=self.DIAGONAL_POSITION_TOLERANCE,
                    orientation_tolerance_deg=1.0,
                )
            )
            stages.append(
                stage(
                    f"대각선 안착 {cycle}/{self.DIAGONAL_SEAT_CYCLES}: 안정화",
                    0.7,
                    seated_origin,
                    seated_R,
                    position_tolerance=self.DIAGONAL_POSITION_TOLERANCE,
                    orientation_tolerance_deg=1.0,
                )
            )

        final_lock_pivot_world = (
            seated_origin + seated_R @ self.UPPER_HOOK_SEAT_LOCAL
        )
        seated_lock_deg = self.LOCK_DEG - self.FINAL_LOCK_ROTATION_DEG
        seated_R = self._gripper_rotation(seated_lock_deg)
        seated_origin = (
            final_lock_pivot_world
            - seated_R @ self.UPPER_HOOK_SEAT_LOCAL
        )
        stages.append(
            stage(
                f"대각선 접촉점에서 안쪽으로 {self.FINAL_LOCK_ROTATION_DEG:.1f}도 최종 잠금",
                2.0,
                seated_origin,
                seated_R,
                pivot_world=final_lock_pivot_world,
                pivot_local=self.UPPER_HOOK_SEAT_LOCAL,
                pivot_start_deg=self.LOCK_DEG,
                pivot_end_deg=seated_lock_deg,
                checkpoint="final_lock_rotation",
                position_tolerance=0.002,
                orientation_tolerance_deg=0.5,
            )
        )
        stages.append(
            stage(
                "최종 하부 훅 잠금 안정화",
                1.0,
                seated_origin,
                seated_R,
                position_tolerance=0.002,
                orientation_tolerance_deg=0.5,
            )
        )

        test_lifted_origin = seated_origin + np.array(
            [0.0, 0.0, self.TEST_LIFT_HEIGHT]
        )
        intermediate_lifted_origin = seated_origin + np.array(
            [0.0, 0.0, self.INTERMEDIATE_LIFT_HEIGHT]
        )
        lifted_origin = seated_origin + np.array(
            [0.0, 0.0, self.LIFT_HEIGHT]
        )
        self.expected_seated_bowl_in_gripper = seated_R.T @ (
            bowl_position - seated_origin
        )
        pour_R = self._gripper_rotation(seated_lock_deg + self.POUR_DEG)

        stages.extend(
            [
                stage(
                    "위·아래 걸림 검사 및 안정화",
                    2.0,
                    seated_origin,
                    seated_R,
                    checkpoint="lock_inspection",
                    position_tolerance=0.002,
                    orientation_tolerance_deg=0.5,
                ),
                stage(
                    "파지 확인용 40mm 시험 상승",
                    2.0,
                    test_lifted_origin,
                    seated_R,
                    checkpoint="test_lift",
                    position_tolerance=0.003,
                    orientation_tolerance_deg=0.7,
                ),
                stage(
                    "중간 높이 150mm 상승",
                    3.0,
                    intermediate_lifted_origin,
                    seated_R,
                    checkpoint="intermediate_lift",
                ),
                stage(
                    "그릇 본 상승",
                    6.0,
                    lifted_origin,
                    seated_R,
                    checkpoint="full_lift",
                ),
                stage("들림 상태 안정화", 1.0, lifted_origin, seated_R),
                stage("손목을 기울여 붓기", 4.0, lifted_origin, pour_R),
                stage("붓는 자세 유지", 2.0, lifted_origin, pour_R),
                stage("손목 원위치", 4.0, lifted_origin, seated_R),
                stage("완료 자세 유지", 1.0, lifted_origin, seated_R),
            ]
        )
        if self.manual_handoff:
            return stages[:1]
        return stages

    def _gripper_rotation(self, wrist_deg):
        return (
            self.bowl_frame
            @ Rotation.from_euler("y", wrist_deg, degrees=True).as_matrix()
        )

    def _pivot_link6_pose(self, stage, alpha):
        wrist_deg = self._lerp(
            stage.pivot_start_deg, stage.pivot_end_deg, alpha
        )
        gripper_rotation = self._gripper_rotation(wrist_deg)
        pivot_world = stage.pivot_world
        if stage.pivot_end_world is not None:
            motion_start = float(stage.pivot_motion_start)
            pivot_alpha = np.clip(
                (alpha - motion_start) / max(1.0 - motion_start, 1.0e-6),
                0.0,
                1.0,
            )
            pivot_alpha = pivot_alpha * pivot_alpha * (
                3.0 - 2.0 * pivot_alpha
            )
            pivot_world = self._lerp(
                stage.pivot_world, stage.pivot_end_world, pivot_alpha
            )
        gripper_position = (
            pivot_world - gripper_rotation @ stage.pivot_local
        )
        return self._link6_pose(gripper_position, gripper_rotation)

    def _link6_pose(self, gripper_position_world, gripper_rotation_world):
        # T_world_gripper = T_world_link6 * T_link6_gripper
        link_rotation = (
            gripper_rotation_world @ self.LINK6_TO_GRIPPER_ROTATION.T
        )
        link_position_world = (
            gripper_position_world
            - link_rotation @ self.LINK6_TO_GRIPPER_POSITION
        )
        link_position_base = link_position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        return link_position_base, link_rotation

    def _validate_goals(self):
        seed = self.stages[0].joint_target.copy()
        print("=== 자동 파지 웨이포인트 IK 검사 ===")
        for index, stage in enumerate(self.stages):
            if stage.pivot_world is not None:
                goals = [
                    self._pivot_link6_pose(stage, alpha)
                    for alpha in np.linspace(0.0, 1.0, 16)
                ]
            else:
                goals = [(stage.position, stage.orientation)]

            max_error_mm = 0.0
            for goal_index, (position, orientation) in enumerate(goals):
                if index == 0 and goal_index == 0:
                    q = stage.joint_target
                    _, _, error_mm = self.arm.solve_ik(
                        position, target_R=orientation, seed_6=q
                    )
                else:
                    q, _, error_mm = self.arm.solve_ik(
                        position, target_R=orientation, seed_6=seed
                    )
                if not np.all(np.isfinite(q)) or error_mm > 8.0:
                    raise RuntimeError(
                        f"자동 단계 '{stage.name}' IK 실패: "
                        f"위치 오차 {error_mm:.2f} mm"
                    )
                seed = q
                max_error_mm = max(max_error_mm, error_mm)
            print(f"  {stage.name}: 최대 오차 {max_error_mm:.2f} mm")

    def _enter_stage(self, index):
        self.stage_index = index
        self.stage_elapsed = 0.0
        self.arrival_wait = 0.0
        self.wait_report_second = 0
        self.contact_stall_time = 0.0
        self.command_position = None
        self.command_orientation = None
        position, orientation = self.arm.current_pose()
        self.stage_start_position = position.astype(np.float64)
        self.stage_start_orientation = orientation.astype(np.float64)
        self.stage_start_joints = self.arm.current_joints().astype(np.float64)
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        self.stage_start_bowl_position = bowl_position
        self.stage_start_gripper_position = gripper_position
        self.stage_start_bowl_in_gripper = gripper_orientation.T @ (
            bowl_position - gripper_position
        )
        self.stage_start_bowl_orientation_in_gripper = (
            gripper_orientation.T @ bowl_orientation
        )
        print(
            f"[자동 {index + 1:02d}/{len(self.stages):02d}] "
            f"{self.stages[index].name}"
        )

    def _finish_stage(self, stage):
        if stage.joint_target is not None:
            # 다음 Cartesian 단계가 같은 elbow-up IK 분기를 이어받는다.
            self.arm._last_q = stage.joint_target.copy()

        if stage.checkpoint == "manual_handoff":
            self.auto_control.capture_current_target()
            self.handoff_ready = True
            self.finished = True
            print(
                "[자동 접근 완료] 그리퍼가 림 접촉 기준보다 180mm 위 "
                "안전 대기점에 "
                "도착했습니다. 키보드 TSC 미세조작으로 전환합니다."
            )
            return

        if (
            stage.checkpoint == "upper_contact_seek"
            and not self.upper_contact_detected
        ):
            self._fail(
                "추가 50mm 하강 범위 안에서 고리 천장 접촉을 감지하지 "
                "못했습니다. 잠금 회전을 시작하지 않습니다."
            )
            return

        if stage.checkpoint == "lock_inspection":
            self._print_grasp_diagnostics(stage)

        lift_checks = {
            "test_lift": (
                "40mm 시험 상승",
                self.TEST_LIFT_HEIGHT,
            ),
            "intermediate_lift": (
                "150mm 중간 상승",
                self.INTERMEDIATE_LIFT_HEIGHT - self.TEST_LIFT_HEIGHT,
            ),
            "full_lift": (
                "450mm 본 상승",
                self.LIFT_HEIGHT - self.INTERMEDIATE_LIFT_HEIGHT,
            ),
        }
        if stage.checkpoint in lift_checks:
            label, expected_rise = lift_checks[stage.checkpoint]
            if not self._validate_lift_follow(stage, label, expected_rise):
                return

        next_index = self.stage_index + 1
        if next_index >= len(self.stages):
            self.finished = True
            print("[자동][완료] 접촉 파지 → 상승 → 붓기 동작을 한 번 수행했습니다.")
            return
        self._enter_stage(next_index)

    def _command_pose_error(self):
        if self.command_position is None or self.command_orientation is None:
            return 0.0, 0.0
        actual_position_world, actual_orientation = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        actual_position = actual_position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        position_error = float(
            np.linalg.norm(self.command_position - actual_position)
        )
        orientation_error = np.degrees(
            Rotation.from_matrix(
                self.command_orientation @ actual_orientation.T
            ).magnitude()
        )
        return position_error, float(orientation_error)

    def _rebase_after_upper_contact(self):
        """실제 천장 접촉점에 맞춰 뒤의 잠금·상승 경로를 평행 이동한다."""
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        actual_pivot_world = (
            gripper_position
            + gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )

        lock_stage_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].pivot_world is not None
        )
        planned_pivot_world = self.stages[lock_stage_index].pivot_world
        translation = actual_pivot_world - planned_pivot_world

        for index in range(self.stage_index + 1, len(self.stages)):
            original = self.stages[index]
            pivot_world = original.pivot_world
            pivot_end_world = original.pivot_end_world
            if pivot_world is not None:
                pivot_world = pivot_world + translation
            if pivot_end_world is not None:
                pivot_end_world = pivot_end_world + translation
            self.stages[index] = replace(
                original,
                position=(original.position + translation).astype(np.float32),
                pivot_world=pivot_world,
                pivot_end_world=pivot_end_world,
            )

        actual_link_position_world, _ = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        actual_link_position = actual_link_position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        descent = max(
            0.0, float(self.stage_start_position[2] - actual_link_position[2])
        )
        print(
            f"[자동][천장 접촉] 실제 추가 하강량 = {descent * 1000:.1f} mm, "
            "이 위치를 기준으로 잠금·상승 경로를 재정렬했습니다."
        )

    def _rebase_after_lower_contact(self):
        """실제 하부 훅 접촉 자세를 기준으로 안착·상승 경로를 재정렬한다."""
        lock_stage = self.stages[self.stage_index]
        translation_mm, orientation_delta_deg = (
            self._rebase_future_from_actual_stage(lock_stage)
        )
        print(
            "[자동][하부 훅 접촉] 실제 접촉 자세를 기준으로 재정렬합니다: "
            f"위치 보정 {translation_mm:.1f} mm, "
            f"자세 보정 {orientation_delta_deg:.2f} deg. "
            f"손목을 고정하고 최대 {self.DIAGONAL_SEAT_CYCLES}회의 "
            "중심·하강 대각선 안착을 시작합니다."
        )

    def _retarget_final_lock_from_current(self):
        """대각선 접촉점에서 남은 이동을 건너뛰고 최종 잠금을 만든다."""
        final_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "final_lock_rotation"
        )
        original_final = self.stages[final_index]
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        actual_wrist_deg = self._wrist_deg(gripper_orientation)
        target_wrist_deg = actual_wrist_deg - self.FINAL_LOCK_ROTATION_DEG
        pivot_world = (
            gripper_position
            + gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        target_gripper_orientation = self._gripper_rotation(target_wrist_deg)
        target_gripper_position = (
            pivot_world
            - target_gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        target_position, target_orientation = self._link6_pose(
            target_gripper_position, target_gripper_orientation
        )

        translation = target_position - original_final.position
        rotation_delta = target_orientation @ original_final.orientation.T
        self.stages[final_index] = replace(
            original_final,
            position=target_position.astype(np.float32),
            orientation=target_orientation.astype(np.float64),
            pivot_world=pivot_world,
            pivot_start_deg=actual_wrist_deg,
            pivot_end_deg=target_wrist_deg,
        )
        for index in range(final_index + 1, len(self.stages)):
            original = self.stages[index]
            self.stages[index] = replace(
                original,
                position=(original.position + translation).astype(np.float32),
                orientation=(rotation_delta @ original.orientation).astype(
                    np.float64
                ),
            )

        bowl_position = self._bowl_position()
        self.expected_seated_bowl_in_gripper = (
            target_gripper_orientation.T
            @ (bowl_position - target_gripper_position)
        )
        print(
            "[자동][대각선 접촉] 남은 대각선 이동을 생략하고 "
            f"현재 접촉점에서 {self.FINAL_LOCK_ROTATION_DEG:.1f}도 "
            "최종 잠금으로 전환합니다."
        )
        self._enter_stage(final_index)

    def _rebase_future_from_actual_stage(self, stage):
        """접촉으로 목표에 못 간 경우 이후 경로를 실제 정지 자세에 맞춘다."""
        actual_position_world, actual_orientation = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        base_offset = np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        actual_position = actual_position_world - base_offset
        translation = actual_position - stage.position
        rotation_delta = actual_orientation @ stage.orientation.T

        actual_gripper_position, actual_gripper_orientation = (
            self._rigid_body_pose(
                self.arm.actor, self.gripper_body_index
            )
        )
        planned_gripper_orientation = (
            stage.orientation @ self.LINK6_TO_GRIPPER_ROTATION
        )
        planned_gripper_position = (
            stage.position
            + base_offset
            + stage.orientation @ self.LINK6_TO_GRIPPER_POSITION
        )
        actual_pivot_world = (
            actual_gripper_position
            + actual_gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        planned_pivot_world = (
            planned_gripper_position
            + planned_gripper_orientation @ self.UPPER_HOOK_SEAT_LOCAL
        )
        pivot_translation = actual_pivot_world - planned_pivot_world
        actual_wrist_deg = self._wrist_deg(actual_gripper_orientation)
        planned_wrist_deg = self._wrist_deg(planned_gripper_orientation)
        pivot_angle_shift = actual_wrist_deg - planned_wrist_deg

        for index in range(self.stage_index + 1, len(self.stages)):
            original = self.stages[index]
            pivot_world = original.pivot_world
            pivot_end_world = original.pivot_end_world
            pivot_start_deg = original.pivot_start_deg
            pivot_end_deg = original.pivot_end_deg
            if pivot_world is not None:
                pivot_world = pivot_world + pivot_translation
                pivot_start_deg = pivot_start_deg + pivot_angle_shift
                pivot_end_deg = pivot_end_deg + pivot_angle_shift
            if pivot_end_world is not None:
                pivot_end_world = pivot_end_world + pivot_translation
            rebased = replace(
                original,
                position=(original.position + translation).astype(np.float32),
                orientation=(rotation_delta @ original.orientation).astype(
                    np.float64
                ),
                pivot_world=pivot_world,
                pivot_end_world=pivot_end_world,
                pivot_start_deg=pivot_start_deg,
                pivot_end_deg=pivot_end_deg,
            )
            if rebased.pivot_world is not None:
                endpoint_position, endpoint_orientation = self._pivot_link6_pose(
                    rebased, 1.0
                )
                rebased = replace(
                    rebased,
                    position=endpoint_position.astype(np.float32),
                    orientation=endpoint_orientation.astype(np.float64),
                )
            self.stages[index] = rebased

        translation_mm = float(np.linalg.norm(translation) * 1000.0)
        orientation_delta_deg = float(
            np.degrees(Rotation.from_matrix(rotation_delta).magnitude())
        )
        return translation_mm, orientation_delta_deg

    def _wrist_deg(self, gripper_orientation):
        relative_rotation = self.bowl_frame.T @ gripper_orientation
        return float(
            np.degrees(
                np.arctan2(
                    relative_rotation[0, 2], relative_rotation[0, 0]
                )
            )
        )

    def _validate_lift_follow(self, stage, label, expected_gripper_rise):
        """상승량뿐 아니라 bowl/gripper 상대 자세가 유지되는지 검사한다."""
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        gripper_rise = float(
            gripper_position[2] - self.stage_start_gripper_position[2]
        )
        bowl_rise = float(
            bowl_position[2] - self.stage_start_bowl_position[2]
        )
        follow_ratio = bowl_rise / max(gripper_rise, 1.0e-6)
        bowl_in_gripper = gripper_orientation.T @ (
            bowl_position - gripper_position
        )
        relative_position_drift = float(
            np.linalg.norm(
                bowl_in_gripper - self.stage_start_bowl_in_gripper
            )
        )
        bowl_orientation_in_gripper = (
            gripper_orientation.T @ bowl_orientation
        )
        relative_orientation_drift = float(
            np.degrees(
                Rotation.from_matrix(
                    bowl_orientation_in_gripper
                    @ self.stage_start_bowl_orientation_in_gripper.T
                ).magnitude()
            )
        )
        print(
            f"[자동][{label} 추종 검사] 그리퍼 {gripper_rise * 1000:.1f} mm, "
            f"그릇 {bowl_rise * 1000:.1f} mm, 추종률 {follow_ratio * 100:.1f}%, "
            f"상대 위치 변화 {relative_position_drift * 1000:.1f} mm, "
            f"상대 자세 변화 {relative_orientation_drift:.2f} deg"
        )

        minimum_gripper_rise = (
            expected_gripper_rise * self.LIFT_MIN_COMMAND_FRACTION
        )
        if gripper_rise < minimum_gripper_rise:
            self._print_grasp_diagnostics(stage)
            self._fail(
                f"{label}에서 그리퍼가 명령 상승량의 "
                f"{self.LIFT_MIN_COMMAND_FRACTION * 100:.0f}%에 도달하지 못했습니다."
            )
            return False
        if (
            follow_ratio < self.LIFT_MIN_FOLLOW_RATIO
            or relative_position_drift
            > self.LIFT_MAX_RELATIVE_POSITION_DRIFT
            or relative_orientation_drift
            > self.LIFT_MAX_RELATIVE_ORIENTATION_DRIFT_DEG
        ):
            self._print_grasp_diagnostics(stage)
            self._fail(
                f"{label}에서 그릇이 그리퍼를 안정적으로 따라오지 않아 "
                "하부 훅 미안착으로 판정합니다."
            )
            return False
        return True

    def _stage_error(self, stage):
        actual_position_world, actual_orientation = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        actual_position = actual_position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        position_error = float(np.linalg.norm(stage.position - actual_position))
        orientation_error = np.degrees(
            Rotation.from_matrix(stage.orientation @ actual_orientation.T).magnitude()
        )
        return position_error, float(orientation_error)

    def _print_pose_diagnostics(self, stage):
        actual_position_world, actual_orientation = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        target_position_world = stage.position + np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        orientation_error = np.degrees(
            Rotation.from_matrix(stage.orientation @ actual_orientation.T).magnitude()
        )
        print("[자동][자세 진단]")
        print(f"  link_6 목표(world) = {np.round(target_position_world, 4)} m")
        print(f"  link_6 실제(world) = {np.round(actual_position_world, 4)} m")
        print(
            f"  위치 오차 = {np.linalg.norm(target_position_world - actual_position_world) * 1000:.1f} mm, "
            f"자세 오차 = {orientation_error:.2f} deg"
        )
        print(f"  실제 관절각 = {np.round(np.rad2deg(self.arm.current_joints()), 1)} deg")
        if self.auto_control.target is not None:
            print(
                "  목표 관절각 = "
                f"{np.round(np.rad2deg(self.auto_control.target), 1)} deg"
            )
            print(
                "  중력보상 토크 = "
                f"{np.round(self.auto_control.last_gravity_torque, 1)} Nm"
            )
            print(
                "  중력보상 목표각 보정 = "
                f"{np.round(np.rad2deg(self.auto_control.last_gravity_bias), 2)} deg"
            )
            print(
                "  정착 목표각 보정 = "
                f"{np.round(np.rad2deg(self.auto_control.last_settle_trim), 2)} deg"
            )
            print(
                "  위치 드라이브 명령각 = "
                f"{np.round(np.rad2deg(self.auto_control.last_drive_target), 1)} deg"
            )

    def _print_grasp_diagnostics(self, stage):
        self._print_pose_diagnostics(stage)
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        bowl_position, _ = self._rigid_body_pose(self.bowl_actor, 0)
        bowl_in_gripper = gripper_orientation.T @ (
            bowl_position - gripper_position
        )
        expected_locked = (
            self.BOWL_FROM_GRIPPER_LOCKED
            if self.expected_seated_bowl_in_gripper is None
            else self.expected_seated_bowl_in_gripper
        )
        locked_error = np.linalg.norm(bowl_in_gripper - expected_locked)
        print("[자동][파지 진단]")
        print(f"  gripper 실제(world) = {np.round(gripper_position, 4)} m")
        print(f"  bowl 실제(world) = {np.round(bowl_position, 4)} m")
        print(f"  gripper 기준 bowl = {np.round(bowl_in_gripper, 4)} m")
        print(
            f"  미세 안착 목표 = {np.round(expected_locked, 4)} m, "
            f"위치 차이 = {locked_error * 1000:.1f} mm"
        )

    def _fail(self, message):
        self.auto_control.capture_current_target()
        self.failed = True
        self.finished = True
        print(f"[자동][실패] {message}")

    def _rigid_body_pose(self, actor, body_index):
        states = self.gym.get_actor_rigid_body_states(
            self.env, actor, gymapi.STATE_POS
        )
        position = states["pose"]["p"][body_index]
        orientation = states["pose"]["r"][body_index]
        position_array = np.array(
            [position["x"], position["y"], position["z"]], dtype=np.float64
        )
        orientation_matrix = Rotation.from_quat(
            [
                orientation["x"],
                orientation["y"],
                orientation["z"],
                orientation["w"],
            ]
        ).as_matrix()
        return position_array, orientation_matrix

    def _bowl_position(self):
        position, _ = self._rigid_body_pose(self.bowl_actor, 0)
        return position

    @staticmethod
    def _lerp(start, goal, alpha):
        return start + (goal - start) * alpha

    @staticmethod
    def _slerp_matrix(start, goal, alpha):
        delta = Rotation.from_matrix(goal @ start.T).as_rotvec()
        return Rotation.from_rotvec(delta * alpha).as_matrix() @ start
