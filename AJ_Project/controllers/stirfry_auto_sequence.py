"""두산 A0509 볶음 그릇의 접촉 기반 자동 파지·붓기 시퀀스.

이 그리퍼는 열고 닫히는 집게가 아니다. 두꺼운 상부 접촉부를 그릇 림
위쪽에 먼저 댄 뒤 손목축을 회전해 아래쪽 훅까지 걸리게 하는 고정형
그리퍼다. 따라서 이 모듈은 bowl actor를 순간이동시키거나 로봇에 가짜로
고정하지 않고, CAD에서 검증한 상·하 접촉 형상과 PhysX 접촉만 사용한다.

대상은 그리퍼가 맞도록 설계된 원본 크기(Ø250 mm)의 조리 그릇이다.
준비 테이블의 재료 그릇들은 0.75배로 축소되어 있어 같은 파지 좌표를
사용할 수 없다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from isaacgym import gymapi


@dataclass(frozen=True)
class _PoseStage:
    name: str
    duration_s: float
    position: np.ndarray
    orientation: np.ndarray


class StirfryAutoSequence:
    """그릇 하나를 실제 접촉으로 걸어 들어 올린 뒤 한 번 기울인다."""

    HOME_Q = np.zeros(6, dtype=np.float32)

    # stirfry_gripper.step / stirfry_bowl.step의 CAD 검증값(m).
    # 손목각 0도인 gripper frame에서 gripper 원점 -> 정지 bowl 원점.
    BOWL_FROM_GRIPPER_ZERO = np.array(
        [0.280512057, 0.0, -0.049234758], dtype=np.float64
    )

    # 3도에서 두꺼운 상부가 먼저 닿고, 1.75도에서 하부까지 걸린다.
    PRELOCK_DEG = 8.0
    UPPER_CONTACT_DEG = 3.0
    LOCK_DEG = 1.75

    APPROACH_DISTANCE = 0.10
    STAGING_HEIGHT = 0.25
    LIFT_HEIGHT = 0.45
    POUR_DEG = 50.0

    # 통합 URDF의 link_6 -> gripper fixed joint.
    LINK6_TO_GRIPPER_POSITION = np.array([0.0, 0.0, 0.016], dtype=np.float64)
    LINK6_TO_GRIPPER_ROTATION = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()

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
        self.stage_start_position = None
        self.stage_start_orientation = None
        self.finished = False
        self.failed = False
        self.initial_bowl_z = None

        self.arm.go_joints(self.HOME_Q)
        print(
            "[자동] 원본 크기 조리 그릇 1개를 대상으로 접촉 기반 파지·붓기를 시작합니다."
        )

    def update(self):
        """시뮬레이션 루프에서 물리 스텝 전에 매 프레임 한 번 호출한다."""
        if self.finished:
            return

        if self.stage_index < 0:
            self._update_home()
            return

        stage = self.stages[self.stage_index]
        alpha = min(1.0, self.stage_elapsed / stage.duration_s)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

        target_position = self._lerp(
            self.stage_start_position, stage.position, alpha
        )
        target_orientation = self._slerp_matrix(
            self.stage_start_orientation, stage.orientation, alpha
        )
        self.arm.go_cartesian(target_position, target_R=target_orientation)

        self.stage_elapsed += self.dt
        if self.stage_elapsed + 1.0e-9 >= stage.duration_s:
            # 마지막 목표를 정확히 한 번 인가한 후 다음 단계로 넘어간다.
            self.arm.go_cartesian(stage.position, target_R=stage.orientation)
            self._finish_stage(stage)

    def _update_home(self):
        self.arm.go_joints(self.HOME_Q)
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
        # bowl은 원형이므로 yaw 자체보다 테이블 슬롯 방향이 중요하다.
        # 이 위치를 유지한 채 8 -> 3 -> 1.75도로 회전하면 상부가 먼저
        # 닿은 후 하부 훅이 걸린다는 CAD clearance 검증을 그대로 따른다.
        gripper_origin = (
            bowl_position
            - self.bowl_frame @ self.BOWL_FROM_GRIPPER_ZERO
        )
        away_from_bowl = -self.bowl_frame[:, 0]
        preinsert_origin = (
            gripper_origin + away_from_bowl * self.APPROACH_DISTANCE
        )
        staging_origin = preinsert_origin + np.array(
            [0.0, 0.0, self.STAGING_HEIGHT]
        )
        lifted_origin = gripper_origin + np.array([0.0, 0.0, self.LIFT_HEIGHT])

        prelock_R = self._gripper_rotation(self.PRELOCK_DEG)
        upper_contact_R = self._gripper_rotation(self.UPPER_CONTACT_DEG)
        lock_R = self._gripper_rotation(self.LOCK_DEG)
        pour_R = self._gripper_rotation(self.LOCK_DEG + self.POUR_DEG)

        def stage(name, duration_s, gripper_position, gripper_orientation):
            link_position, link_orientation = self._link6_pose(
                gripper_position, gripper_orientation
            )
            return _PoseStage(
                name,
                duration_s,
                link_position.astype(np.float32),
                link_orientation.astype(np.float64),
            )

        return [
            stage("슬롯 위 안전 위치", 4.0, staging_origin, prelock_R),
            stage("슬롯 입구로 하강", 3.0, preinsert_origin, prelock_R),
            stage("그릇 쪽으로 수평 진입", 2.5, gripper_origin, prelock_R),
            stage("두꺼운 상부를 그릇 위에 접촉", 2.0, gripper_origin, upper_contact_R),
            stage("상부 접촉 안정화", 1.0, gripper_origin, upper_contact_R),
            stage("손목 회전으로 하부 훅 잠금", 2.0, gripper_origin, lock_R),
            stage("위·아래 걸림 안정화", 1.5, gripper_origin, lock_R),
            stage("그릇 들어 올리기", 4.0, lifted_origin, lock_R),
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
        seed = self.arm.current_joints().copy()
        print("=== 자동 파지 웨이포인트 IK 검사 ===")
        for stage in self.stages:
            q, _, error_mm = self.arm.solve_ik(
                stage.position, target_R=stage.orientation, seed_6=seed
            )
            if not np.all(np.isfinite(q)) or error_mm > 8.0:
                raise RuntimeError(
                    f"자동 단계 '{stage.name}' IK 실패: 위치 오차 {error_mm:.2f} mm"
                )
            seed = q
            print(f"  {stage.name}: 오차 {error_mm:.2f} mm")

        # 실제 실행 seed는 현재 관절에서 다시 시작해야 한다.
        self.arm._last_q = self.arm.current_joints().copy()

    def _enter_stage(self, index):
        self.stage_index = index
        self.stage_elapsed = 0.0
        position, orientation = self.arm.current_pose()
        self.stage_start_position = position.astype(np.float64)
        self.stage_start_orientation = orientation.astype(np.float64)
        print(
            f"[자동 {index + 1:02d}/{len(self.stages):02d}] "
            f"{self.stages[index].name}"
        )

    def _finish_stage(self, stage):
        if stage.name == "그릇 들어 올리기":
            lifted = float(self._bowl_position()[2] - self.initial_bowl_z)
            print(f"[자동] 실제 그릇 상승량 = {lifted:.3f} m")
            if lifted < 0.20:
                self.failed = True
                self.finished = True
                print(
                    "[자동][실패] 그릇이 충분히 들리지 않아 붓기를 중단합니다. "
                    "접촉 간극 또는 물리 파라미터를 확인하세요."
                )
                return

        next_index = self.stage_index + 1
        if next_index >= len(self.stages):
            self.finished = True
            print("[자동][완료] 접촉 파지 → 상승 → 붓기 동작을 한 번 수행했습니다.")
            return
        self._enter_stage(next_index)

    def _bowl_position(self):
        states = self.gym.get_actor_rigid_body_states(
            self.env, self.bowl_actor, gymapi.STATE_POS
        )
        position = states["pose"]["p"][0]
        return np.array(
            [position["x"], position["y"], position["z"]], dtype=np.float64
        )

    @staticmethod
    def _lerp(start, goal, alpha):
        return start + (goal - start) * alpha

    @staticmethod
    def _slerp_matrix(start, goal, alpha):
        delta = Rotation.from_matrix(goal @ start.T).as_rotvec()
        return Rotation.from_rotvec(delta * alpha).as_matrix() @ start
