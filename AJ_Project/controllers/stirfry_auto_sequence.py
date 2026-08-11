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

from dataclasses import dataclass

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
    pivot_local: object = None
    pivot_start_deg: object = None
    pivot_end_deg: object = None


class StirfryAutoSequence:
    """그릇 하나를 실제 접촉으로 걸어 들어 올린 뒤 한 번 기울인다."""

    HOME_Q = np.zeros(6, dtype=np.float32)

    # stirfry_gripper.step / stirfry_bowl.step의 깊은 고리 잠금 자세 CAD 값(m).
    # 빨간 원으로 확인한 안쪽 천장 접점을 유지한 채 아래 훅이 그릇 바닥에
    # 닿기 직전인 -6.54도 자세에서의 bowl origin (gripper frame)이다.
    BOWL_FROM_GRIPPER_LOCKED = np.array(
        [0.244150678, 0.0, -0.048588184], dtype=np.float64
    )

    # 림보다 10 mm 위에서 잠금 반대 방향으로 88도까지 열어 입구 접촉을
    # 피한다. 그 높이를 유지한 채 그릇 중심 방향으로 36 mm 삽입한 다음
    # 총 16 mm 하강한다(진입 여유 10 mm + 안착 6 mm). 그러면 입구 면이
    # 아니라 STEP face 15(빨간 원의 깊은 안쪽 천장)와 림 윗면이 약
    # 0.48 mm 간격으로 마주한다. 그 접점을 유지해 -6.54도까지 회전하면
    # 아래 훅 간격은 약 0.013 mm다.
    VERTICAL_ENTRY_DEG = 70.0
    UPPER_OPEN_DEG = 88.0
    LOCK_DEG = -6.54

    VERTICAL_STAGING_CLEARANCE = 0.18
    UPPER_PRECONTACT_CLEARANCE = 0.04
    UPPER_INSERT_CLEARANCE = 0.010
    UPPER_INSERT_DISTANCE = 0.036
    UPPER_SEAT_DESCENT = 0.016
    TEST_LIFT_HEIGHT = 0.04
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
    TEST_LIFT_MIN_RISE = 0.015
    FULL_LIFT_MIN_RISE = 0.30

    # 통합 URDF의 link_6 -> gripper fixed joint.
    LINK6_TO_GRIPPER_POSITION = np.array([0.0, 0.0, 0.016], dtype=np.float64)
    LINK6_TO_GRIPPER_ROTATION = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()

    # CAD 좌표(m): 최초 림 정렬 기준과 깊은 고리 안쪽 천장 피벗축.
    # 두 번째 값은 입구 쪽 face들을 제외하고 STEP face 15와 bowl rim top
    # face의 최근접점만으로 계산했다.
    UPPER_ENTRY_REFERENCE_LOCAL = np.array(
        [0.1545, 0.0, 0.0355], dtype=np.float64
    )
    UPPER_HOOK_SEAT_LOCAL = np.array(
        [0.128614516, 0.0, 0.045266161], dtype=np.float64
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
    ):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.arm = arm
        self.bowl_actor = bowl_actor
        self.base_z = float(base_z)
        self.dt = float(dt)

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
        self.finished = False
        self.failed = False
        self.initial_bowl_z = None

        self.arm.go_joints(self.HOME_Q)
        self.auto_control = StirfryAutoJointControl(self.arm, dt=self.dt)
        self.auto_control.command(self.HOME_Q)
        print(
            "[자동] 원본 크기 조리 그릇 1개를 대상으로 접촉 기반 파지·붓기를 시작합니다."
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

        if self.stage_elapsed + 1.0e-9 < stage.duration_s:
            self.stage_elapsed += self.dt
            return

        # 예정 시간이 지난 뒤에는 실제 rigid body가 목표에 도착할 때까지
        # 최종 명령을 유지한다. 로그의 단계 번호만 진행하는 상황을 막는다.
        position_error, orientation_error = self._stage_error(stage)
        if (
            position_error <= self.POSITION_TOLERANCE
            and orientation_error <= self.ORIENTATION_TOLERANCE_DEG
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
        if self.arrival_wait >= self.ARRIVAL_TIMEOUT_S:
            self._print_pose_diagnostics(stage)
            self._fail(
                f"'{stage.name}' 실제 목표 미도착: 위치 {position_error * 1000:.1f} mm, "
                f"자세 {orientation_error:.2f} deg"
            )

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
        self.initial_bowl_z = float(bowl_position[2])
        self.stages = self._build_stages(bowl_position)
        self._validate_goals()
        print(
            "[자동] 그릇 기준점 = "
            f"{np.round(bowl_position, 4)} m, 상부부터 접촉하도록 진입합니다."
        )
        self._enter_stage(0)

    def _build_stages(self, bowl_position):
        # 림보다 10 mm 위에서 88도로 고리 입구를 크게 연다. 이 여유 높이를
        # 유지한 채 36 mm 중심 방향으로 통과시킨 뒤 총 16 mm 하강해 림을
        # 빨간 원의 깊은 안쪽 천장(face 15)까지 보낸다. 이후 이 접촉축을
        # 고정한 원호로 -6.54도까지 회전하면 아래 훅도 그릇 바닥에 닿는다.
        outer_rim_world = (
            bowl_position + self.bowl_frame @ self.BOWL_NEAR_RIM_LOCAL
        )
        entry_R = self._gripper_rotation(self.VERTICAL_ENTRY_DEG)
        open_R = self._gripper_rotation(self.UPPER_OPEN_DEG)
        lock_R = self._gripper_rotation(self.LOCK_DEG)
        entry_origin = (
            outer_rim_world
            - entry_R @ self.UPPER_ENTRY_REFERENCE_LOCAL
        )
        open_origin = (
            entry_origin
            + self.bowl_frame[:, 2] * self.UPPER_INSERT_CLEARANCE
        )
        inserted_origin = (
            open_origin
            + self.bowl_frame[:, 0] * self.UPPER_INSERT_DISTANCE
        )
        seated_origin = (
            inserted_origin
            - self.bowl_frame[:, 2] * self.UPPER_SEAT_DESCENT
        )
        upper_seat_pivot_world = (
            seated_origin + open_R @ self.UPPER_HOOK_SEAT_LOCAL
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
        test_lifted_origin = locked_origin + np.array(
            [0.0, 0.0, self.TEST_LIFT_HEIGHT]
        )
        lifted_origin = locked_origin + np.array([0.0, 0.0, self.LIFT_HEIGHT])

        pour_R = self._gripper_rotation(self.LOCK_DEG + self.POUR_DEG)

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

        return [
            stage(
                "그릇 위 수직 파지 대기(elbow-up)",
                4.0,
                staging_origin,
                entry_R,
                joint_target=staging_q,
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
                "림 위 10mm 여유를 유지하며 중심으로 36mm 삽입",
                4.0,
                inserted_origin,
                open_R,
            ),
            stage(
                "림 윗면을 깊은 고리 안쪽 천장으로 16mm 하강",
                4.0,
                seated_origin,
                open_R,
            ),
            stage(
                "깊은 상부 고리 안쪽 림 안착 안정화",
                2.0,
                seated_origin,
                open_R,
            ),
            stage(
                "깊은 상부 천장 접점을 유지하며 하부 훅 회전 잠금",
                8.0,
                locked_origin,
                lock_R,
                pivot_world=upper_seat_pivot_world,
                pivot_local=self.UPPER_HOOK_SEAT_LOCAL,
                pivot_start_deg=self.UPPER_OPEN_DEG,
                pivot_end_deg=self.LOCK_DEG,
            ),
            stage(
                "위·아래 걸림 검사 및 안정화",
                3.0,
                locked_origin,
                lock_R,
                checkpoint="lock_inspection",
            ),
            stage(
                "파지 확인용 4cm 시험 상승",
                2.0,
                test_lifted_origin,
                lock_R,
                checkpoint="test_lift",
            ),
            stage(
                "그릇 본 상승",
                4.0,
                lifted_origin,
                lock_R,
                checkpoint="full_lift",
            ),
            stage("들림 상태 안정화", 1.0, lifted_origin, lock_R),
            stage("손목을 기울여 붓기", 4.0, lifted_origin, pour_R),
            stage("붓는 자세 유지", 2.0, lifted_origin, pour_R),
            stage("손목 원위치", 4.0, lifted_origin, lock_R),
            stage("완료 자세 유지", 1.0, lifted_origin, lock_R),
        ]

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
        gripper_position = (
            stage.pivot_world
            - gripper_rotation @ stage.pivot_local
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
        position, orientation = self.arm.current_pose()
        self.stage_start_position = position.astype(np.float64)
        self.stage_start_orientation = orientation.astype(np.float64)
        self.stage_start_joints = self.arm.current_joints().astype(np.float64)
        print(
            f"[자동 {index + 1:02d}/{len(self.stages):02d}] "
            f"{self.stages[index].name}"
        )

    def _finish_stage(self, stage):
        if stage.joint_target is not None:
            # 다음 Cartesian 단계가 같은 elbow-up IK 분기를 이어받는다.
            self.arm._last_q = stage.joint_target.copy()

        if stage.checkpoint == "lock_inspection":
            self._print_grasp_diagnostics(stage)

        if stage.checkpoint == "test_lift":
            lifted = float(self._bowl_position()[2] - self.initial_bowl_z)
            print(f"[자동][시험 상승] 실제 그릇 상승량 = {lifted:.3f} m")
            if lifted < self.TEST_LIFT_MIN_RISE:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "4cm 시험 상승에서 그릇이 따라오지 않았습니다. "
                    "파지 위치 근처에서 정지합니다."
                )
                return

        if stage.checkpoint == "full_lift":
            lifted = float(self._bowl_position()[2] - self.initial_bowl_z)
            print(f"[자동][본 상승] 실제 그릇 상승량 = {lifted:.3f} m")
            if lifted < self.FULL_LIFT_MIN_RISE:
                self._fail("본 상승 중 그릇을 놓쳐 붓기를 중단합니다.")
                return

        next_index = self.stage_index + 1
        if next_index >= len(self.stages):
            self.finished = True
            print("[자동][완료] 접촉 파지 → 상승 → 붓기 동작을 한 번 수행했습니다.")
            return
        self._enter_stage(next_index)

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
        locked_error = np.linalg.norm(
            bowl_in_gripper - self.BOWL_FROM_GRIPPER_LOCKED
        )
        print("[자동][파지 진단]")
        print(f"  gripper 실제(world) = {np.round(gripper_position, 4)} m")
        print(f"  bowl 실제(world) = {np.round(bowl_position, 4)} m")
        print(f"  gripper 기준 bowl = {np.round(bowl_in_gripper, 4)} m")
        print(
            f"  CAD 잠금 기준 = {np.round(self.BOWL_FROM_GRIPPER_LOCKED, 4)} m, "
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
