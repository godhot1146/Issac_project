"""상부 고리 접촉점을 피벗으로 사용하는 볶음 그릇 자동 파지 시퀀스.

성공한 TSC 기록에서 상부 고리와 림의 실제 접촉 위치 및 손목 회전 범위를
추출했다. 그리퍼를 림에 수직 안착한 뒤 손목 관절만 돌리지 않고, 팔 전체의
IK를 매 프레임 다시 풀어 상부 접촉면을 따라 link_6를 원호 이동한다. 회전
중에는 현재 그릇 자세와 PhysX 접촉 패치를 추종하고, 접촉이 끊기면 회전을
멈춘 채 조금 하강해 재접촉한다. 공용 키보드 텔레오프는 수정하지 않는다.
"""

from dataclasses import replace

import numpy as np
from isaacgym import gymapi

from stirfry_auto_sequence import StirfryAutoSequence, _PoseStage


class StirfryRecordedGraspSequence(StirfryAutoSequence):
    """접촉점 기준의 연속 피벗으로 그릇을 걸고 약 180 mm 상승한다."""

    # 39-part 성공 TRACE에서 최초 안정 접촉 당시의 near rim을 그리퍼
    # 좌표로 역산한 값이다. 기존 CAD 천장점 [0.150, 0, 0.031]보다 실제
    # 접촉은 약 17 mm 안쪽, 13 mm 아래에서 형성됐다. 이 값은 최초 접촉
    # 탐색에만 쓴다. 별도 수평 삽입 없이 수직으로 접촉면을 2 mm 더 눌러
    # 안정화한 뒤 PhysX 접촉 패치를 연속 회전의 피벗으로 사용한다.
    RECORDED_UPPER_CONTACT_LOCAL = np.array(
        [0.1334, 0.0, 0.0184], dtype=np.float64
    )

    # 성공 TRACE의 최초 안정 접촉 자세와 최종 잠금 자세.
    PIVOT_START_DEG = 59.3868
    PIVOT_END_DEG = -4.6149

    PRECONTACT_CLEARANCE_M = 0.010
    CONTACT_SEEK_PENETRATION_M = 0.004
    UPPER_SEAT_PRELOAD_M = 0.002
    MIN_UPPER_SEAT_CONTACTS = 2
    UPPER_SEAT_CONFIRM_S = 0.50
    PIVOT_DURATION_S = 2.25
    PIVOT_ROLLING_PRELOAD_M = 0.006
    PIVOT_RECOVERY_MAX_DESCENT_M = 0.010
    PIVOT_RECOVERY_DESCENT_SPEED_M_S = 0.008
    PIVOT_CONTACT_LOSS_GRACE_S = 0.05
    PIVOT_CONTACT_RECOVERY_HOLD_S = 0.10
    PIVOT_CONTACT_FAILURE_S = 1.50
    PIVOT_TOTAL_TIMEOUT_S = 14.0
    PIVOT_LOCK_COMMIT_PROGRESS = 0.50
    UPPER_CONTACT_TRACK_RADIUS_M = 0.025
    UPPER_CONTACT_CAPTURE_CLUSTER_M = 0.006
    PIVOT_CARTESIAN_CORRECTION_MAX_M = 0.012
    PIVOT_CARTESIAN_CORRECTION_ALPHA = 0.35
    DIRECT_LIFT_COMMAND_M = 0.140

    MIN_LOCK_LIFT_M = 0.015
    MIN_FINAL_LIFT_M = 0.150
    MAX_PRELOCK_TILT_DEG = 8.0
    MAX_LOCK_TILT_DEG = 15.0
    CONTACT_CONFIRM_S = 0.05

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
        self.recorded_contact_time = 0.0
        self.recorded_upper_gripper_local = None
        self.recorded_upper_bowl_local = None
        self.recorded_contact_positions_available = None
        self.recorded_pivot_progress = 0.0
        self.recorded_pivot_wall_time = 0.0
        self.recorded_pivot_contact_loss_time = 0.0
        self.recorded_pivot_contact_stable_time = 0.0
        self.recorded_pivot_contact_ready = True
        self.recorded_pivot_lock_committed = False
        self.recorded_pivot_recovery_descent = 0.0
        self.recorded_pivot_cartesian_correction = np.zeros(
            3, dtype=np.float64
        )
        self.recorded_pivot_anchor_gap = np.zeros(3, dtype=np.float64)
        self.recorded_pivot_last_report_second = -1
        super().__init__(
            gym,
            sim,
            env,
            arm,
            bowl_actor,
            base_z=base_z,
            dt=dt,
            slot_direction_xy=slot_direction_xy,
            # 부모의 검증된 180 mm 안전 접근 한 단계만 재사용한다.
            manual_handoff=True,
        )

        self.gripper_env_body = gym.get_actor_rigid_body_index(
            env,
            arm.actor,
            self.gripper_body_index,
            gymapi.DOMAIN_ENV,
        )
        self.bowl_env_body = gym.get_actor_rigid_body_index(
            env, bowl_actor, 0, gymapi.DOMAIN_ENV
        )
        print(
            "[접촉 피벗 자동 파지] 안전 접근 뒤 상부 고리 접촉점을 "
            "축으로 팔 전체를 연속 이동합니다."
        )
        print(
            "[접촉 피벗 자동 파지] "
            "수평 삽입 없이 수직 안착·2 mm 면 밀착 후, "
            f"손목 {self.PIVOT_START_DEG:.1f} -> "
            f"{self.PIVOT_END_DEG:.1f} deg 동안 실제 접촉면을 추종합니다."
        )

    def _build_stages(self, bowl_position):
        stages = list(super()._build_stages(bowl_position))
        if len(stages) != 1 or stages[0].checkpoint != "manual_handoff":
            raise RuntimeError("접촉 피벗의 안전 접근 단계를 만들지 못했습니다.")
        # 비접촉 고공 접근만 25% 단축한다. 충돌 여유가 큰 구간이라
        # 접촉 안착이나 잠금 회전의 안정성에는 영향을 주지 않는다.
        stages[0] = replace(stages[0], duration_s=3.0)

        bowl_position = np.asarray(bowl_position, dtype=np.float64)
        self.recorded_initial_bowl_position = bowl_position.copy()
        _, self.recorded_initial_bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )

        up = self.bowl_frame[:, 2]
        near_rim_world = (
            bowl_position + self.bowl_frame @ self.BOWL_NEAR_RIM_LOCAL
        )
        start_R = self._gripper_rotation(self.PIVOT_START_DEG)
        locked_R = self._gripper_rotation(self.PIVOT_END_DEG)

        precontact_pivot = (
            near_rim_world + up * self.PRECONTACT_CLEARANCE_M
        )
        seek_pivot = (
            near_rim_world - up * self.CONTACT_SEEK_PENETRATION_M
        )

        def link_pose_for_pivot(pivot_world, gripper_R):
            gripper_position = (
                pivot_world
                - gripper_R @ self.RECORDED_UPPER_CONTACT_LOCAL
            )
            return self._link6_pose(gripper_position, gripper_R)

        def pose_stage(
            name,
            duration_s,
            position,
            orientation,
            checkpoint="",
            position_tolerance=0.005,
            orientation_tolerance_deg=1.0,
        ):
            return _PoseStage(
                name=name,
                duration_s=float(duration_s),
                position=np.asarray(position, dtype=np.float32),
                orientation=np.asarray(orientation, dtype=np.float64),
                checkpoint=checkpoint,
                position_tolerance=position_tolerance,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )

        precontact_position, precontact_R = link_pose_for_pivot(
            precontact_pivot, start_R
        )
        seek_position, seek_R = link_pose_for_pivot(seek_pivot, start_R)
        preloaded_position = seek_position - up * self.UPPER_SEAT_PRELOAD_M

        pivot_start_world = near_rim_world.copy()
        locked_position, locked_link_R = link_pose_for_pivot(
            pivot_start_world, locked_R
        )
        pivot_stage = _PoseStage(
            name="상부 실제 접촉면 추종 회전·하부 훅 잠금",
            duration_s=self.PIVOT_DURATION_S,
            position=locked_position.astype(np.float32),
            orientation=locked_link_R.astype(np.float64),
            checkpoint="recorded_pivot_lock",
            pivot_world=pivot_start_world,
            pivot_end_world=None,
            pivot_local=self.RECORDED_UPPER_CONTACT_LOCAL,
            pivot_start_deg=self.PIVOT_START_DEG,
            pivot_end_deg=self.PIVOT_END_DEG,
            position_tolerance=0.015,
            orientation_tolerance_deg=5.0,
        )

        lifted_position = (
            locked_position + up * self.DIRECT_LIFT_COMMAND_M
        )

        stages.extend(
            [
                pose_stage(
                    "상부 고리 접촉면을 림 10mm 위로 정렬",
                    1.2,
                    precontact_position,
                    precontact_R,
                ),
                pose_stage(
                    "상부 고리 접촉면을 림까지 저속 안착",
                    1.0,
                    seek_position,
                    seek_R,
                    checkpoint="recorded_contact_seek",
                    position_tolerance=0.001,
                    orientation_tolerance_deg=0.8,
                ),
                pose_stage(
                    "첫 접촉 뒤 상부 접촉면을 2mm 수직 밀착",
                    0.5,
                    preloaded_position,
                    seek_R,
                    checkpoint="recorded_upper_preload",
                    position_tolerance=0.0005,
                    orientation_tolerance_deg=0.8,
                ),
                pivot_stage,
                pose_stage(
                    "잠금 성공 뒤 그릇 140mm 즉시 수직 상승",
                    2.0,
                    lifted_position,
                    locked_link_R,
                    checkpoint="recorded_final_lift",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                pose_stage(
                    "접촉 피벗 자동 파지 완료 자세 유지",
                    0.25,
                    lifted_position,
                    locked_link_R,
                    checkpoint="recorded_complete",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
            ]
        )
        return stages

    def update(self):
        if (
            not self.finished
            and self.stage_index >= 0
            and self.stages[self.stage_index].checkpoint
            == "recorded_pivot_lock"
        ):
            self._update_contact_following_pivot(
                self.stages[self.stage_index]
            )
            return

        if not self.finished and self.stage_index >= 0:
            stage = self.stages[self.stage_index]
            if stage.checkpoint in {
                "recorded_contact_seek",
                "recorded_upper_preload",
            }:
                contact_count = self._gripper_bowl_contact_count()
                minimum_contacts = (
                    self.MIN_UPPER_SEAT_CONTACTS
                    if stage.checkpoint == "recorded_upper_preload"
                    else 1
                )
                if contact_count >= minimum_contacts:
                    self.recorded_contact_time += self.dt
                else:
                    self.recorded_contact_time = 0.0
                confirmation_time = (
                    self.UPPER_SEAT_CONFIRM_S
                    if stage.checkpoint == "recorded_upper_preload"
                    else self.CONTACT_CONFIRM_S
                )
                if (
                    self.recorded_contact_time >= confirmation_time
                    and (
                        stage.checkpoint != "recorded_upper_preload"
                        or self.stage_elapsed + 1.0e-9 >= stage.duration_s
                    )
                ):
                    position_error, orientation_error = self._stage_error(stage)
                    print(
                        f"[자동][조건 충족] {stage.name}: "
                        f"그리퍼-그릇 접촉 {contact_count}개가 "
                        f"{self.recorded_contact_time:.2f}s 유지됨 "
                        "(실제 자세 오차: 위치 "
                        f"{position_error * 1000:.1f} mm, 자세 "
                        f"{orientation_error:.2f} deg)"
                    )
                    self._finish_stage(stage)
                    return
        super().update()

    def _enter_stage(self, index):
        super()._enter_stage(index)
        if self.stages[index].checkpoint == "recorded_pivot_lock":
            self.recorded_pivot_progress = 0.0
            self.recorded_pivot_wall_time = 0.0
            self.recorded_pivot_contact_loss_time = 0.0
            self.recorded_pivot_contact_stable_time = 0.0
            self.recorded_pivot_contact_ready = True
            self.recorded_pivot_lock_committed = False
            self.recorded_pivot_recovery_descent = 0.0
            self.recorded_pivot_cartesian_correction[:] = 0.0
            self.recorded_pivot_anchor_gap[:] = 0.0
            self.recorded_pivot_last_report_second = -1

    def _stage_completion_override(
        self, stage, position_error, orientation_error
    ):
        if stage.checkpoint == "recorded_contact_seek":
            count = self._gripper_bowl_contact_count()
            if count > 0:
                return f"그리퍼-그릇 접촉 {count}개 확인"
            return None

        if stage.checkpoint == "recorded_upper_preload":
            count = self._gripper_bowl_contact_count()
            if (
                count >= self.MIN_UPPER_SEAT_CONTACTS
                and self.recorded_contact_time >= self.UPPER_SEAT_CONFIRM_S
            ):
                return (
                    f"접촉 {count}개가 {self.recorded_contact_time:.2f}s "
                    "유지된 상태로 2mm 밀착"
                )
            return None

        bowl_lift = self._recorded_bowl_lift()
        if stage.checkpoint == "recorded_pivot_lock":
            if bowl_lift >= self.MIN_LOCK_LIFT_M:
                return (
                    f"자연 잠금 회전으로 그릇 {bowl_lift * 1000:.1f} mm "
                    "상승 확인"
                )
            return None

        if stage.checkpoint in {"recorded_final_lift", "recorded_complete"}:
            if bowl_lift >= self.MIN_FINAL_LIFT_M:
                return f"그릇 총 상승량 {bowl_lift * 1000:.1f} mm 확인"
        return None

    def _gripper_bowl_contacts(self):
        contacts = []
        for contact in self.gym.get_env_rigid_contacts(self.env):
            bodies = {int(contact["body0"]), int(contact["body1"])}
            if bodies == {self.gripper_env_body, self.bowl_env_body}:
                contacts.append(contact)
        return contacts

    def _gripper_bowl_contact_count(self):
        return len(self._gripper_bowl_contacts())

    @staticmethod
    def _contact_field(contact, *names):
        for name in names:
            try:
                return contact[name]
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            try:
                return getattr(contact, name)
            except AttributeError:
                pass
        return None

    @staticmethod
    def _vec3_array(value):
        if value is None:
            return None
        try:
            return np.array(
                [value["x"], value["y"], value["z"]],
                dtype=np.float64,
            )
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            return np.array(
                [value.x, value.y, value.z], dtype=np.float64
            )
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        return vector[:3] if vector.size >= 3 else None

    def _upper_contact_manifold(
        self,
        reference_gripper_local=None,
        capture_nearest_cluster=False,
    ):
        """상부 고리 접촉 패치의 두 물체 로컬 중심과 개수를 반환한다."""
        contacts = self._gripper_bowl_contacts()
        candidates = []
        position_field_seen = False

        for contact in contacts:
            gripper_is_body0 = (
                int(contact["body0"]) == self.gripper_env_body
            )
            gripper_value = self._contact_field(
                contact,
                "localPos0" if gripper_is_body0 else "localPos1",
                "local_pos0" if gripper_is_body0 else "local_pos1",
            )
            bowl_value = self._contact_field(
                contact,
                "localPos1" if gripper_is_body0 else "localPos0",
                "local_pos1" if gripper_is_body0 else "local_pos0",
            )
            gripper_local = self._vec3_array(gripper_value)
            bowl_local = self._vec3_array(bowl_value)
            if gripper_local is None or bowl_local is None:
                continue
            position_field_seen = True
            force = self._contact_field(contact, "lambda")
            try:
                weight = max(abs(float(force)), 1.0e-6)
            except (TypeError, ValueError):
                weight = 1.0
            candidates.append((gripper_local, bowl_local, weight))

        if candidates and reference_gripper_local is not None:
            if capture_nearest_cluster:
                # 첫 접촉 전체의 힘 가중 평균은 고리 끝·윗면까지 섞어
                # 성공 기록점에서 20 mm 이상 벗어났다. 성공 기록점에
                # 가장 가까운 접촉을 먼저 고르고 그 주변 패치만 사용한다.
                nearest_gripper = min(
                    candidates,
                    key=lambda item: np.linalg.norm(
                        item[0] - reference_gripper_local
                    ),
                )[0]
                candidates = [
                    item
                    for item in candidates
                    if np.linalg.norm(item[0] - nearest_gripper)
                    <= self.UPPER_CONTACT_CAPTURE_CLUSTER_M
                ]
            else:
                candidates = [
                    item
                    for item in candidates
                    if np.linalg.norm(
                        item[0] - reference_gripper_local
                    )
                    <= self.UPPER_CONTACT_TRACK_RADIUS_M
                ]

        if candidates:
            self.recorded_contact_positions_available = True
            gripper_points = np.asarray(
                [item[0] for item in candidates], dtype=np.float64
            )
            bowl_points = np.asarray(
                [item[1] for item in candidates], dtype=np.float64
            )
            weights = np.asarray(
                [item[2] for item in candidates], dtype=np.float64
            )
            weights = np.asarray(weights, dtype=np.float64)
            weights /= np.sum(weights)
            return (
                np.sum(gripper_points * weights[:, None], axis=0),
                np.sum(bowl_points * weights[:, None], axis=0),
                len(candidates),
            )

        if contacts and not position_field_seen:
            if self.recorded_contact_positions_available is not False:
                print(
                    "[접촉 피벗][접촉 좌표 경고] localPos0/1 필드를 "
                    "읽지 못해 접촉 개수와 이동 림 기준점으로 추종합니다."
                )
            self.recorded_contact_positions_available = False
            return None, None, len(contacts)
        return None, None, 0

    def _retarget_recorded_preload_from_contact(self):
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        target_gripper_position = (
            gripper_position
            - self.bowl_frame[:, 2] * self.UPPER_SEAT_PRELOAD_M
        )
        target_position, target_orientation = self._link6_pose(
            target_gripper_position, gripper_orientation
        )
        preload_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "recorded_upper_preload"
        )
        self.stages[preload_index] = replace(
            self.stages[preload_index],
            position=target_position.astype(np.float32),
            orientation=target_orientation.astype(np.float64),
        )
        print(
            "[접촉 피벗][상부 면 밀착 목표] 첫 접촉 자세에서 수직 "
            f"아래로 {self.UPPER_SEAT_PRELOAD_M * 1000:.0f} mm 추가 하강, "
            f"접촉 {self.MIN_UPPER_SEAT_CONTACTS}개 이상을 "
            f"{self.UPPER_SEAT_CONFIRM_S:.2f}s 유지"
        )

    def _current_near_rim_world(self):
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        orientation_delta = (
            bowl_orientation @ self.recorded_initial_bowl_orientation.T
        )
        initial_rim_offset = (
            self.bowl_frame @ self.BOWL_NEAR_RIM_LOCAL
        )
        return bowl_position + orientation_delta @ initial_rim_offset

    def _rebase_after_recorded_seat(self):
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        gripper_local, bowl_local, contact_count = (
            self._upper_contact_manifold(
                self.RECORDED_UPPER_CONTACT_LOCAL,
                capture_nearest_cluster=True,
            )
        )
        if gripper_local is not None and bowl_local is not None:
            gripper_contact_world = (
                gripper_position + gripper_orientation @ gripper_local
            )
            bowl_contact_world = (
                bowl_position + bowl_orientation @ bowl_local
            )
            actual_pivot_world = 0.5 * (
                gripper_contact_world + bowl_contact_world
            )
            contact_source = "PhysX 상부 접촉 패치"
        else:
            actual_pivot_world = self._current_near_rim_world()
            contact_source = "이동 림 기준점 대체값"

        # 두 물체에서 같은 월드 점을 가리키도록 로컬 좌표를 다시 만든다.
        # 이후 그릇이 움직여도 bowl local 점을 월드로 변환해 피벗을 추종한다.
        actual_gripper_local = gripper_orientation.T @ (
            actual_pivot_world - gripper_position
        )
        actual_bowl_local = bowl_orientation.T @ (
            actual_pivot_world - bowl_position
        )
        self.recorded_upper_gripper_local = actual_gripper_local.copy()
        self.recorded_upper_bowl_local = actual_bowl_local.copy()
        capture_offset = np.linalg.norm(
            actual_gripper_local - self.RECORDED_UPPER_CONTACT_LOCAL
        )
        actual_wrist_deg = self._wrist_deg(gripper_orientation)

        pivot_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "recorded_pivot_lock"
        )
        original = self.stages[pivot_index]
        pivot_translation = actual_pivot_world - original.pivot_world
        wrist_shift_deg = actual_wrist_deg - original.pivot_start_deg
        rebased = replace(
            original,
            pivot_world=actual_pivot_world,
            pivot_end_world=None,
            pivot_local=actual_gripper_local,
            pivot_start_deg=original.pivot_start_deg + wrist_shift_deg,
            pivot_end_deg=original.pivot_end_deg + wrist_shift_deg,
        )
        endpoint_position, endpoint_orientation = self._pivot_link6_pose(
            rebased, 1.0
        )
        endpoint_translation = endpoint_position - original.position
        endpoint_rotation = endpoint_orientation @ original.orientation.T
        self.stages[pivot_index] = replace(
            rebased,
            position=endpoint_position.astype(np.float32),
            orientation=endpoint_orientation.astype(np.float64),
        )

        for index in range(pivot_index + 1, len(self.stages)):
            stage = self.stages[index]
            self.stages[index] = replace(
                stage,
                position=(stage.position + endpoint_translation).astype(
                    np.float32
                ),
                orientation=(endpoint_rotation @ stage.orientation).astype(
                    np.float64
                ),
            )
        print(
            "[접촉 피벗][실제 상부 접촉 패치 설정] "
            f"{contact_source}, 접촉 {contact_count}개, "
            f"계획 대비 {np.linalg.norm(pivot_translation) * 1000:.1f} mm, "
            f"기록 접점 대비 {capture_offset * 1000:.1f} mm, "
            f"손목 {wrist_shift_deg:+.2f} deg 보정, "
            "그리퍼 로컬 접점 "
            f"{np.round(actual_gripper_local * 1000, 1)} mm"
        )

    def _command_contact_following_pivot(self, stage):
        """현재 그릇 접촉점을 따라가며 팔 전체의 피벗 자세를 명령한다."""
        alpha = self.recorded_pivot_progress
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        wrist_deg = self._lerp(
            stage.pivot_start_deg, stage.pivot_end_deg, smooth_alpha
        )
        gripper_orientation = self._gripper_rotation(wrist_deg)

        if self.recorded_upper_bowl_local is not None:
            bowl_position, bowl_orientation = self._rigid_body_pose(
                self.bowl_actor, 0
            )
            pivot_world = (
                bowl_position
                + bowl_orientation @ self.recorded_upper_bowl_local
            )
        else:
            pivot_world = self._current_near_rim_world()

        rolling_preload = (
            self.PIVOT_ROLLING_PRELOAD_M * smooth_alpha
        )
        total_descent = (
            rolling_preload + self.recorded_pivot_recovery_descent
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        gripper_position = (
            pivot_world
            - gripper_orientation @ self.recorded_upper_gripper_local
            + self.recorded_pivot_cartesian_correction
            - world_up * total_descent
        )
        target_position, target_orientation = self._link6_pose(
            gripper_position, gripper_orientation
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
                f"'{stage.name}' 이동 중 IK 실패: "
                f"위치 오차 {ik_error_mm:.2f} mm"
            )
            return
        self.arm._last_q = target_joints.copy()
        self.auto_control.command(
            target_joints, settling=alpha >= 1.0
        )

    def _update_pivot_anchor_correction(self):
        """두 고정 접점의 실제 3D 간격을 다음 팔 목표에 폐루프 반영한다."""
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        bowl_position, bowl_orientation = self._rigid_body_pose(
            self.bowl_actor, 0
        )
        actual_gripper_anchor = (
            gripper_position
            + gripper_orientation @ self.recorded_upper_gripper_local
        )
        actual_bowl_anchor = (
            bowl_position
            + bowl_orientation @ self.recorded_upper_bowl_local
        )
        anchor_gap = actual_bowl_anchor - actual_gripper_anchor
        correction_gap = anchor_gap.copy()
        # 양의 Z 간격은 의도적으로 그리퍼를 림보다 조금 낮게 눌러 생긴다.
        # 이를 위쪽 보정으로 상쇄하지 않고, 수평 오차와 고리가 림보다
        # 위로 들린 경우(음의 Z)만 Cartesian 보정한다.
        correction_gap[2] = min(correction_gap[2], 0.0)
        gap_norm = float(np.linalg.norm(correction_gap))
        if gap_norm > self.PIVOT_CARTESIAN_CORRECTION_MAX_M:
            correction_target = (
                correction_gap
                * self.PIVOT_CARTESIAN_CORRECTION_MAX_M
                / gap_norm
            )
        else:
            correction_target = correction_gap
        self.recorded_pivot_anchor_gap = anchor_gap
        self.recorded_pivot_cartesian_correction = self._lerp(
            self.recorded_pivot_cartesian_correction,
            correction_target,
            self.PIVOT_CARTESIAN_CORRECTION_ALPHA,
        )

    def _update_contact_following_pivot(self, stage):
        """전반은 상부 접촉을 복구하고 후반은 하부 잠금까지 커밋한다."""
        bowl_tilt_deg = self._recorded_bowl_tilt_deg()
        if bowl_tilt_deg > self.MAX_LOCK_TILT_DEG:
            self._print_grasp_diagnostics(stage)
            self._fail(
                "접촉점 피벗 회전 중 그릇 기울기가 "
                f"{bowl_tilt_deg:.2f}도로 커졌습니다."
            )
            return

        _, _, contact_count = (
            self._upper_contact_manifold(
                self.recorded_upper_gripper_local
            )
        )
        raw_contact_count = self._gripper_bowl_contact_count()
        self._update_pivot_anchor_correction()

        if (
            not self.recorded_pivot_lock_committed
            and self.recorded_pivot_progress
            >= self.PIVOT_LOCK_COMMIT_PROGRESS
        ):
            self.recorded_pivot_lock_committed = True
            self.recorded_pivot_contact_ready = True
            self.recorded_pivot_contact_loss_time = 0.0
            print(
                "[접촉 피벗][잠금 커밋] 회전 "
                f"{self.recorded_pivot_progress * 100:.0f}%부터는 "
                "상부 접촉이 하부 훅 접촉으로 전환되어도 멈추지 않고 "
                "최종 잠금 자세까지 회전합니다."
            )

        if self.recorded_pivot_lock_committed:
            # 성공 기록에서도 회전 후반에는 상부 접촉 패치가 하부 훅과
            # 옆면 접촉으로 전환된다. 이 구간은 상부 필터가 0이어도
            # 실패가 아니며, 남은 회전을 끝내야 실제 잠금이 형성된다.
            self.recorded_pivot_contact_ready = True
            if contact_count == 0:
                self.recorded_pivot_recovery_descent = min(
                    self.PIVOT_RECOVERY_MAX_DESCENT_M,
                    self.recorded_pivot_recovery_descent
                    + self.PIVOT_RECOVERY_DESCENT_SPEED_M_S * self.dt,
                )
        elif contact_count > 0:
            self.recorded_pivot_contact_loss_time = 0.0
            self.recorded_pivot_contact_stable_time += self.dt
            if (
                not self.recorded_pivot_contact_ready
                and self.recorded_pivot_contact_stable_time
                >= self.PIVOT_CONTACT_RECOVERY_HOLD_S
            ):
                self.recorded_pivot_contact_ready = True
                print(
                    "[접촉 피벗][재접촉] 상부 접촉이 "
                    f"{self.recorded_pivot_contact_stable_time:.2f}s "
                    "유지되어 회전을 재개합니다."
                )
        else:
            self.recorded_pivot_contact_stable_time = 0.0
            self.recorded_pivot_contact_loss_time += self.dt
            if (
                self.recorded_pivot_contact_ready
                and self.recorded_pivot_contact_loss_time
                >= self.PIVOT_CONTACT_LOSS_GRACE_S
            ):
                self.recorded_pivot_contact_ready = False
                print(
                    "[접촉 피벗][접촉 이탈] 회전을 일시 정지하고 "
                    "수직 하강으로 상부 접촉을 복구합니다."
                )
            if not self.recorded_pivot_contact_ready:
                self.recorded_pivot_recovery_descent = min(
                    self.PIVOT_RECOVERY_MAX_DESCENT_M,
                    self.recorded_pivot_recovery_descent
                    + self.PIVOT_RECOVERY_DESCENT_SPEED_M_S * self.dt,
                )

        if self.recorded_pivot_lock_committed or (
            self.recorded_pivot_contact_ready and contact_count > 0
        ):
            self.recorded_pivot_progress = min(
                1.0,
                self.recorded_pivot_progress
                + self.dt / self.PIVOT_DURATION_S,
            )
        self.recorded_pivot_wall_time += self.dt
        self.stage_elapsed = (
            self.recorded_pivot_progress * self.PIVOT_DURATION_S
        )
        self._command_contact_following_pivot(stage)
        if self.finished:
            return

        report_second = int(self.recorded_pivot_wall_time)
        if report_second > self.recorded_pivot_last_report_second:
            self.recorded_pivot_last_report_second = report_second
            commanded_descent = (
                self.PIVOT_ROLLING_PRELOAD_M
                * self.recorded_pivot_progress
                + self.recorded_pivot_recovery_descent
            )
            print(
                "[접촉 피벗][추종] "
                f"회전 {self.recorded_pivot_progress * 100:.0f}%, "
                f"상부/전체 접촉 {contact_count}/{raw_contact_count}개, "
                "수직 보정 "
                f"{commanded_descent * 1000:.1f} mm, "
                "접점 간격 "
                f"{np.linalg.norm(self.recorded_pivot_anchor_gap) * 1000:.1f} mm, "
                "3D 보정 "
                f"{np.linalg.norm(self.recorded_pivot_cartesian_correction) * 1000:.1f} mm"
            )

        if (
            not self.recorded_pivot_lock_committed
            and not self.recorded_pivot_contact_ready
            and self.recorded_pivot_contact_loss_time
            >= self.PIVOT_CONTACT_FAILURE_S
            and self.recorded_pivot_recovery_descent
            >= self.PIVOT_RECOVERY_MAX_DESCENT_M - 1.0e-9
        ):
            self._print_grasp_diagnostics(stage)
            self._fail(
                "상부 접촉이 끊긴 뒤 10mm 수직 하강과 3D 접점 "
                "오차 보정 범위에서도 "
                "재접촉하지 못했습니다."
            )
            return

        if self.recorded_pivot_wall_time >= self.PIVOT_TOTAL_TIMEOUT_S:
            self._print_grasp_diagnostics(stage)
            self._fail(
                "접촉 피벗 잠금이 "
                f"{self.PIVOT_TOTAL_TIMEOUT_S:.0f}초 안에 잠금 조건을 "
                "충족하지 못했습니다."
            )
            return

        if (
            self.recorded_pivot_progress >= 1.0
            and raw_contact_count > 0
            and self._recorded_bowl_lift() >= self.MIN_LOCK_LIFT_M
        ):
            position_error, orientation_error = self._stage_error(stage)
            print(
                f"[자동][조건 충족] {stage.name}: "
                f"최종 그리퍼-그릇 접촉 {raw_contact_count}개로 그릇 "
                f"{self._recorded_bowl_lift() * 1000:.1f} mm 상승 "
                "(실제 자세 오차: 위치 "
                f"{position_error * 1000:.1f} mm, 자세 "
                f"{orientation_error:.2f} deg)"
            )
            self._finish_stage(stage)

    def _retarget_direct_lift_from_current(self):
        """잠금 실제 자세에서 다른 동작 없이 월드 Z로 바로 상승한다."""
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        target_gripper_position = gripper_position + np.array(
            [0.0, 0.0, self.DIRECT_LIFT_COMMAND_M], dtype=np.float64
        )
        target_position, target_orientation = self._link6_pose(
            target_gripper_position, gripper_orientation
        )
        lift_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "recorded_final_lift"
        )
        for index in range(lift_index, len(self.stages)):
            self.stages[index] = replace(
                self.stages[index],
                position=target_position.astype(np.float32),
                orientation=target_orientation.astype(np.float64),
            )
        print(
            "[접촉 피벗][직접 상승 목표] 잠금 실제 자세에서 "
            f"월드 Z축으로 {self.DIRECT_LIFT_COMMAND_M * 1000:.0f} mm "
            "즉시 상승"
        )

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
            self.arm._last_q = stage.joint_target.copy()
            self.handoff_ready = False
            print(
                "[접촉 피벗 자동 파지] 180mm 안전 접근 완료. "
                "상부 접촉면 정렬을 시작합니다."
            )
            self._enter_stage(self.stage_index + 1)
            return

        if stage.checkpoint == "recorded_contact_seek":
            contact_count = self._gripper_bowl_contact_count()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            if contact_count <= 0:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 고리 접촉면과 그릇 림의 실제 접촉을 확인하지 "
                    "못했습니다."
                )
                return
            if bowl_tilt_deg > self.MAX_PRELOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 접촉 안착 중 그릇 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커졌습니다."
                )
                return
            print(
                "[접촉 피벗][상부 안착] "
                f"접촉 {contact_count}개, 그릇 기울기 {bowl_tilt_deg:.2f} deg"
            )
            self._retarget_recorded_preload_from_contact()
            self.recorded_contact_time = 0.0

        if stage.checkpoint == "recorded_upper_preload":
            contact_count = self._gripper_bowl_contact_count()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            if contact_count < self.MIN_UPPER_SEAT_CONTACTS:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 접촉면 추가 밀착 뒤 접촉이 "
                    f"{self.MIN_UPPER_SEAT_CONTACTS}개보다 적어 피벗을 "
                    "설정하지 않습니다."
                )
                return
            if self.recorded_contact_time < self.UPPER_SEAT_CONFIRM_S:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 접촉면이 "
                    f"{self.UPPER_SEAT_CONFIRM_S:.2f}s 연속 유지되지 않아 "
                    "피벗을 설정하지 않습니다."
                )
                return
            if bowl_tilt_deg > self.MAX_PRELOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 접촉면 추가 밀착 중 그릇 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커졌습니다."
                )
                return
            print(
                "[접촉 피벗][상부 면 밀착 완료] "
                f"접촉 {contact_count}개를 "
                f"{self.recorded_contact_time:.2f}s 연속 유지, "
                f"그릇 기울기 {bowl_tilt_deg:.2f} deg"
            )
            self._rebase_after_recorded_seat()

        if stage.checkpoint == "recorded_pivot_lock":
            bowl_lift = self._recorded_bowl_lift()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            print(
                "[접촉 피벗][잠금 검사] "
                f"그릇 상승 {bowl_lift * 1000:.1f} mm, "
                f"기울기 {bowl_tilt_deg:.2f} deg"
            )
            if bowl_lift < self.MIN_LOCK_LIFT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "피벗 잠금 뒤 그릇 상승이 15mm보다 작아 하부 고리 "
                    "미안착으로 판정합니다."
                )
                return
            self._retarget_direct_lift_from_current()

        if stage.checkpoint == "recorded_final_lift":
            if not self._validate_lift_follow(
                stage, "140mm 직접 수직 상승", self.DIRECT_LIFT_COMMAND_M
            ):
                return

        if stage.checkpoint == "recorded_complete":
            bowl_lift = self._recorded_bowl_lift()
            if bowl_lift < self.MIN_FINAL_LIFT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "완료 시점의 그릇 총 상승량이 150mm보다 작습니다."
                )
                return
            self.finished = True
            print(
                "[접촉 피벗 자동 파지][완료] "
                f"최종 상승량={bowl_lift * 1000:.1f} mm, "
                f"기울기={self._recorded_bowl_tilt_deg():.2f} deg"
            )
            return

        super()._finish_stage(stage)

    def _print_pose_diagnostics(self, stage):
        diagnostic_stage = stage
        if (
            stage.checkpoint == "recorded_pivot_lock"
            and self.command_position is not None
            and self.command_orientation is not None
        ):
            diagnostic_stage = replace(
                stage,
                position=np.asarray(
                    self.command_position, dtype=np.float32
                ),
                orientation=np.asarray(
                    self.command_orientation, dtype=np.float64
                ),
            )
        super()._print_pose_diagnostics(diagnostic_stage)
        if self.recorded_initial_bowl_position is not None:
            print(
                "  그릇 총 상승량 = "
                f"{self._recorded_bowl_lift() * 1000:.1f} mm"
            )
            print(
                "  그릇 초기 대비 기울기 = "
                f"{self._recorded_bowl_tilt_deg():.2f} deg"
            )
        if stage.checkpoint == "recorded_pivot_lock":
            print(
                "  상부 고정 접점 3D 간격 = "
                f"{np.linalg.norm(self.recorded_pivot_anchor_gap) * 1000:.1f} mm"
            )
            print(
                "  팔 전체 Cartesian 접점 보정 = "
                f"{np.round(self.recorded_pivot_cartesian_correction * 1000, 1)} mm"
            )
