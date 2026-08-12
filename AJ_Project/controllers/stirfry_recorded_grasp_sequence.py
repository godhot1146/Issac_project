"""상부 고리 접촉점을 피벗으로 사용하는 볶음 그릇 자동 파지 시퀀스.

성공한 TSC 기록에서 상부 고리와 림의 실제 접촉 위치 및 손목 회전 범위를
추출했다. 그리퍼를 림에 안착한 뒤 손목 관절만 돌리지 않고, 팔 전체의 IK를
매 프레임 다시 풀어 상부 접촉점이 움직이지 않도록 link_6를 원호 이동한다.
하부 고리가 걸리는 회전 후반에는 피벗축도 위로 이동해 그릇을 자연스럽게
들어 올린다. 공용 키보드 텔레오프는 수정하거나 사용하지 않는다.
"""

from dataclasses import replace

import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation

from stirfry_auto_sequence import StirfryAutoSequence, _PoseStage


class StirfryRecordedGraspSequence(StirfryAutoSequence):
    """접촉점 기준의 연속 피벗으로 그릇을 걸고 약 180 mm 상승한다."""

    # 39-part 성공 TRACE에서 최초 안정 접촉 당시의 near rim을 그리퍼
    # 좌표로 역산한 값이다. 기존 CAD 천장점 [0.150, 0, 0.031]보다 실제
    # 접촉은 약 17 mm 안쪽, 13 mm 아래에서 형성됐다. 이 값은 최초 접촉
    # 탐색에만 쓴다. 7 mm 추가 삽입하고 수직 재접촉한 뒤 실제 림 위치를
    # 그리퍼 로컬로 다시 투영해 연속 회전의 피벗점을 갱신한다.
    RECORDED_UPPER_CONTACT_LOCAL = np.array(
        [0.1334, 0.0, 0.0184], dtype=np.float64
    )

    # 성공 TRACE의 최초 안정 접촉 자세와 최종 잠금 자세.
    PIVOT_START_DEG = 59.3868
    PIVOT_LIFT_START_DEG = 14.8858
    PIVOT_END_DEG = -4.6149
    PIVOT_LIFT_START_ALPHA = (
        (PIVOT_START_DEG - PIVOT_LIFT_START_DEG)
        / (PIVOT_START_DEG - PIVOT_END_DEG)
    )

    PRECONTACT_CLEARANCE_M = 0.010
    CONTACT_SEEK_PENETRATION_M = 0.004
    UPPER_INSERT_DEPTH_M = 0.007
    MIN_UPPER_INSERT_M = 0.005
    UPPER_RESEAT_MAX_DESCENT_M = 0.012
    LOCK_PIVOT_RISE_M = 0.045
    TEST_LIFT_M = 0.040
    TRANSFER_X_M = 0.050
    FINAL_LIFT_COMMAND_M = 0.100

    MIN_LOCK_LIFT_M = 0.015
    MIN_FINAL_LIFT_M = 0.150
    MAX_PRELOCK_TILT_DEG = 8.0
    MAX_LOCK_TILT_DEG = 15.0
    CONTACT_CONFIRM_S = 0.05
    TRANSFER_MAX_RELATIVE_POSITION_DRIFT_M = 0.015
    TRANSFER_MAX_RELATIVE_ORIENTATION_DRIFT_DEG = 8.0

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
            f"첫 접촉 뒤 안쪽으로 {self.UPPER_INSERT_DEPTH_M * 1000:.0f} mm "
            "추가 삽입 후 수직 재안착, "
            f"손목 {self.PIVOT_START_DEG:.1f} -> "
            f"{self.PIVOT_END_DEG:.1f} deg, 후반 피벗 상승 "
            f"{self.LOCK_PIVOT_RISE_M * 1000:.0f} mm"
        )

    def _build_stages(self, bowl_position):
        stages = list(super()._build_stages(bowl_position))
        if len(stages) != 1 or stages[0].checkpoint != "manual_handoff":
            raise RuntimeError("접촉 피벗의 안전 접근 단계를 만들지 못했습니다.")

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
        seek_gripper_position = (
            seek_pivot - start_R @ self.RECORDED_UPPER_CONTACT_LOCAL
        )
        inserted_gripper_position = (
            seek_gripper_position
            + self.bowl_frame[:, 0] * self.UPPER_INSERT_DEPTH_M
        )
        inserted_position, inserted_R = self._link6_pose(
            inserted_gripper_position, start_R
        )
        reseated_position = (
            inserted_position - up * self.UPPER_RESEAT_MAX_DESCENT_M
        )

        pivot_start_world = near_rim_world.copy()
        pivot_end_world = (
            pivot_start_world + up * self.LOCK_PIVOT_RISE_M
        )
        locked_position, locked_link_R = link_pose_for_pivot(
            pivot_end_world, locked_R
        )
        pivot_stage = _PoseStage(
            name="상부 접촉점 기준 연속 회전·후반 상승 잠금",
            duration_s=4.5,
            position=locked_position.astype(np.float32),
            orientation=locked_link_R.astype(np.float64),
            checkpoint="recorded_pivot_lock",
            pivot_world=pivot_start_world,
            pivot_end_world=pivot_end_world,
            pivot_local=self.RECORDED_UPPER_CONTACT_LOCAL,
            pivot_start_deg=self.PIVOT_START_DEG,
            pivot_end_deg=self.PIVOT_END_DEG,
            pivot_motion_start=self.PIVOT_LIFT_START_ALPHA,
            position_tolerance=0.015,
            orientation_tolerance_deg=5.0,
        )

        test_lift_position = locked_position + up * self.TEST_LIFT_M
        transfer_position = test_lift_position + np.array(
            [self.TRANSFER_X_M, 0.0, 0.0], dtype=np.float64
        )
        final_position = (
            transfer_position + up * self.FINAL_LIFT_COMMAND_M
        )

        stages.extend(
            [
                pose_stage(
                    "상부 고리 접촉면을 림 10mm 위로 정렬",
                    2.0,
                    precontact_position,
                    precontact_R,
                ),
                pose_stage(
                    "상부 고리 접촉면을 림까지 저속 안착",
                    1.5,
                    seek_position,
                    seek_R,
                    checkpoint="recorded_contact_seek",
                    position_tolerance=0.001,
                    orientation_tolerance_deg=0.8,
                ),
                pose_stage(
                    "상부 고리를 그릇 안쪽으로 7mm 삽입",
                    1.2,
                    inserted_position,
                    inserted_R,
                    checkpoint="recorded_upper_insert",
                    position_tolerance=0.003,
                    orientation_tolerance_deg=1.0,
                ),
                pose_stage(
                    "삽입 자세에서 상부 접촉면까지 수직 재하강",
                    1.5,
                    reseated_position,
                    inserted_R,
                    checkpoint="recorded_upper_reseat",
                    position_tolerance=0.001,
                    orientation_tolerance_deg=0.8,
                ),
                pivot_stage,
                pose_stage(
                    "파지 검증용 40mm 시험 상승",
                    1.5,
                    test_lift_position,
                    locked_link_R,
                    checkpoint="recorded_test_lift",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                pose_stage(
                    "파지 유지 상태로 X축 50mm 이동",
                    1.5,
                    transfer_position,
                    locked_link_R,
                    checkpoint="recorded_transfer",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                pose_stage(
                    "그릇 100mm 최종 상승",
                    2.0,
                    final_position,
                    locked_link_R,
                    checkpoint="recorded_final_lift",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
                pose_stage(
                    "접촉 피벗 자동 파지 완료 자세 유지",
                    0.5,
                    final_position,
                    locked_link_R,
                    checkpoint="recorded_complete",
                    position_tolerance=0.008,
                    orientation_tolerance_deg=2.0,
                ),
            ]
        )
        return stages

    def update(self):
        if not self.finished and self.stage_index >= 0:
            stage = self.stages[self.stage_index]
            if stage.checkpoint in {
                "recorded_contact_seek",
                "recorded_upper_reseat",
            }:
                contact_count = self._gripper_bowl_contact_count()
                if contact_count > 0:
                    self.recorded_contact_time += self.dt
                else:
                    self.recorded_contact_time = 0.0
                if self.recorded_contact_time >= self.CONTACT_CONFIRM_S:
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

        if (
            not self.finished
            and self.stage_index >= 0
            and self.stages[self.stage_index].checkpoint
            == "recorded_pivot_lock"
            and self._recorded_bowl_tilt_deg() > self.MAX_LOCK_TILT_DEG
        ):
            stage = self.stages[self.stage_index]
            self._print_grasp_diagnostics(stage)
            self._fail(
                "접촉점 피벗 회전 중 그릇 기울기가 "
                f"{self._recorded_bowl_tilt_deg():.2f}도로 커졌습니다."
            )
            return
        super().update()

    def _stage_completion_override(
        self, stage, position_error, orientation_error
    ):
        if stage.checkpoint in {
            "recorded_contact_seek",
            "recorded_upper_reseat",
        }:
            count = self._gripper_bowl_contact_count()
            if count > 0:
                return f"그리퍼-그릇 접촉 {count}개 확인"
            return None

        if stage.checkpoint == "recorded_upper_insert":
            advance = self._recorded_inward_advance()
            if advance >= self.MIN_UPPER_INSERT_M:
                return (
                    f"그릇 대비 안쪽으로 {advance * 1000:.1f} mm 삽입"
                )
            return None

        bowl_lift = self._recorded_bowl_lift()
        if stage.checkpoint == "recorded_pivot_lock":
            if bowl_lift >= self.MIN_LOCK_LIFT_M:
                return (
                    f"후반 피벗 상승으로 그릇 {bowl_lift * 1000:.1f} mm "
                    "추종 확인"
                )
            return None

        if stage.checkpoint in {"recorded_final_lift", "recorded_complete"}:
            if bowl_lift >= self.MIN_FINAL_LIFT_M:
                return f"그릇 총 상승량 {bowl_lift * 1000:.1f} mm 확인"
        return None

    def _gripper_bowl_contact_count(self):
        count = 0
        for contact in self.gym.get_env_rigid_contacts(self.env):
            bodies = {int(contact["body0"]), int(contact["body1"])}
            if bodies == {self.gripper_env_body, self.bowl_env_body}:
                count += 1
        return count

    def _retarget_recorded_insertion_from_contact(self):
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        target_gripper_position = (
            gripper_position
            + self.bowl_frame[:, 0] * self.UPPER_INSERT_DEPTH_M
        )
        target_position, target_orientation = self._link6_pose(
            target_gripper_position, gripper_orientation
        )
        insertion_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "recorded_upper_insert"
        )
        self.stages[insertion_index] = replace(
            self.stages[insertion_index],
            position=target_position.astype(np.float32),
            orientation=target_orientation.astype(np.float64),
        )
        print(
            "[접촉 피벗][상부 삽입 목표] 실제 첫 접촉 자세에서 "
            f"그릇 중심 방향 {self.UPPER_INSERT_DEPTH_M * 1000:.0f} mm"
        )

    def _retarget_recorded_reseat_from_insertion(self):
        gripper_position, gripper_orientation = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        target_gripper_position = (
            gripper_position
            - self.bowl_frame[:, 2] * self.UPPER_RESEAT_MAX_DESCENT_M
        )
        target_position, target_orientation = self._link6_pose(
            target_gripper_position, gripper_orientation
        )
        reseat_index = next(
            index
            for index in range(self.stage_index + 1, len(self.stages))
            if self.stages[index].checkpoint == "recorded_upper_reseat"
        )
        self.stages[reseat_index] = replace(
            self.stages[reseat_index],
            position=target_position.astype(np.float32),
            orientation=target_orientation.astype(np.float64),
        )
        print(
            "[접촉 피벗][상부 재접촉 목표] 삽입 완료 자세에서 수직 아래로 "
            f"최대 {self.UPPER_RESEAT_MAX_DESCENT_M * 1000:.0f} mm"
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
        actual_pivot_world = self._current_near_rim_world()
        actual_pivot_local = (
            gripper_orientation.T
            @ (actual_pivot_world - gripper_position)
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
            pivot_world=original.pivot_world + pivot_translation,
            pivot_end_world=original.pivot_end_world + pivot_translation,
            pivot_local=actual_pivot_local,
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
            "[접촉 피벗][삽입 후 실제 림 피벗 설정] "
            f"계획 대비 {np.linalg.norm(pivot_translation) * 1000:.1f} mm, "
            f"손목 {wrist_shift_deg:+.2f} deg 보정, "
            "그리퍼 로컬 접점 "
            f"{np.round(actual_pivot_local * 1000, 1)} mm"
        )

    def _recorded_inward_advance(self):
        if (
            self.stage_start_gripper_position is None
            or self.stage_start_bowl_position is None
        ):
            return 0.0
        gripper_position, _ = self._rigid_body_pose(
            self.arm.actor, self.gripper_body_index
        )
        bowl_position = self._bowl_position()
        relative_motion = (
            gripper_position - self.stage_start_gripper_position
            - (bowl_position - self.stage_start_bowl_position)
        )
        return float(
            np.dot(
                relative_motion,
                self.bowl_frame[:, 0],
            )
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
            self._retarget_recorded_insertion_from_contact()

        if stage.checkpoint == "recorded_upper_insert":
            contact_count = self._gripper_bowl_contact_count()
            inward_advance = self._recorded_inward_advance()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            print(
                "[접촉 피벗][상부 삽입 검사] "
                f"그릇 대비 안쪽 이동 {inward_advance * 1000:.1f} mm, "
                f"접촉 {contact_count}개, "
                f"그릇 기울기 {bowl_tilt_deg:.2f} deg"
            )
            if inward_advance < self.MIN_UPPER_INSERT_M:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 고리가 그릇 대비 안쪽으로 "
                    f"{self.MIN_UPPER_INSERT_M * 1000:.0f}mm 이상 들어가지 "
                    "못해 수직 재접촉을 시작하지 않습니다."
                )
                return
            if bowl_tilt_deg > self.MAX_PRELOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 고리 삽입 중 그릇 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커졌습니다."
                )
                return
            if contact_count <= 0:
                print(
                    "[접촉 피벗][상부 삽입] 수평 삽입으로 접촉이 "
                    "분리되었습니다. 수직 재하강으로 다시 안착합니다."
                )
            self._retarget_recorded_reseat_from_insertion()
            self.recorded_contact_time = 0.0

        if stage.checkpoint == "recorded_upper_reseat":
            contact_count = self._gripper_bowl_contact_count()
            bowl_tilt_deg = self._recorded_bowl_tilt_deg()
            if contact_count <= 0:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    f"{self.UPPER_RESEAT_MAX_DESCENT_M * 1000:.0f}mm "
                    "수직 재하강 범위에서 상부 고리 접촉면과 "
                    "그릇 림의 재접촉을 확인하지 못했습니다."
                )
                return
            if bowl_tilt_deg > self.MAX_PRELOCK_TILT_DEG:
                self._print_grasp_diagnostics(stage)
                self._fail(
                    "상부 수직 재접촉 중 그릇 기울기가 "
                    f"{bowl_tilt_deg:.2f}도로 커졌습니다."
                )
                return
            print(
                "[접촉 피벗][상부 재안착] "
                f"접촉 {contact_count}개, 그릇 기울기 {bowl_tilt_deg:.2f} deg"
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
            translation_mm, orientation_delta_deg = (
                self._rebase_future_from_actual_stage(stage)
            )
            print(
                "[접촉 피벗][잠금 재정렬] 실제 정지 자세 기준: "
                f"위치 {translation_mm:.1f} mm, "
                f"자세 {orientation_delta_deg:.2f} deg 보정"
            )

        if stage.checkpoint == "recorded_test_lift":
            if not self._validate_lift_follow(
                stage, "40mm 시험 상승", self.TEST_LIFT_M
            ):
                return

        if stage.checkpoint == "recorded_transfer":
            if not self._validate_transfer_retention(stage):
                return

        if stage.checkpoint == "recorded_final_lift":
            if not self._validate_lift_follow(
                stage, "100mm 최종 상승", self.FINAL_LIFT_COMMAND_M
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
            "[접촉 피벗][수평 이동 추종 검사] "
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
                "수평 이동 중 그릇과 그리퍼의 상대 자세가 크게 변했습니다."
            )
            return False
        return True
