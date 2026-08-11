"""볶음 자동 시퀀스 전용 위치 목표 중력보상.

공용 ``DoosanController``는 수정하지 않는다. 기존 PhysX 위치 드라이브를
그대로 사용하되, ``setup_osc()``가 확보한 텐서로 중력 토크를 계산하고
``tau_g / stiffness``만큼 드라이브 목표각을 보정한다.
"""

import numpy as np


class StirfryAutoJointControl:
    """내부 위치제어의 안정성을 유지하며 중력에 의한 정상상태 오차를 줄인다."""

    MAX_BIAS_DEG = 10.0

    def __init__(self, arm):
        if not arm._osc_ready:
            raise RuntimeError(
                "StirfryAutoJointControl 생성 전에 arm.setup_osc()가 필요합니다."
            )

        self.arm = arm
        self.gym = arm.gym
        self.sim = arm.sim
        self.env = arm.env

        dof_properties = self.gym.get_actor_dof_properties(
            self.env, self.arm.actor
        )
        self.stiffness = np.asarray(
            dof_properties["stiffness"], dtype=np.float32
        ).copy()
        self.lower_limits = np.asarray(
            dof_properties["lower"], dtype=np.float32
        ).copy()
        self.upper_limits = np.asarray(
            dof_properties["upper"], dtype=np.float32
        ).copy()
        expected_shape = (self.arm.num_dofs,)
        if (
            self.stiffness.shape != expected_shape
            or not np.all(np.isfinite(self.stiffness))
            or np.any(self.stiffness <= 0.0)
        ):
            raise RuntimeError(
                "자동 중력보상에 필요한 position stiffness를 읽지 못했습니다."
            )
        if (
            self.lower_limits.shape != expected_shape
            or self.upper_limits.shape != expected_shape
            or not np.all(np.isfinite(self.lower_limits))
            or not np.all(np.isfinite(self.upper_limits))
        ):
            raise RuntimeError("자동 제어에 필요한 관절 한계를 읽지 못했습니다.")

        self.max_bias = np.deg2rad(self.MAX_BIAS_DEG)
        self.target = None
        self.last_gravity_torque = np.zeros(self.arm.num_dofs, dtype=np.float32)
        self.last_position_bias = np.zeros(self.arm.num_dofs, dtype=np.float32)
        self.last_drive_target = np.zeros(self.arm.num_dofs, dtype=np.float32)

        print(
            "[자동 제어] 위치 목표 중력보상 활성화 "
            f"(stiffness={np.round(self.stiffness, 1)} Nm/rad, "
            f"보정 한계=±{self.MAX_BIAS_DEG:.0f} deg)"
        )

    def command(self, target_joints):
        """새 실제 목표각을 저장하고 중력 보정된 위치 목표를 인가한다."""
        target = np.asarray(target_joints, dtype=np.float32)
        if (
            target.shape != (self.arm.num_dofs,)
            or not np.all(np.isfinite(target))
        ):
            raise ValueError(
                f"target_joints는 유효한 {self.arm.num_dofs}축 값이어야 합니다."
            )
        self.target = target.copy()
        self.update()

    def capture_current_target(self):
        """실패 지점에서 더 밀지 않도록 현재 관절각을 새 유지 목표로 삼는다."""
        self.target = self.arm.current_joints().astype(np.float32)

    def update(self):
        """저장한 실제 목표각을 한 프레임 중력 보정해 위치제어로 전달한다."""
        if self.target is None:
            raise RuntimeError("자동 관절 목표가 설정되지 않았습니다.")

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        gravity = self.arm._gravity_torque().detach().cpu().numpy()

        # PhysX 위치 드라이브가 만드는 정적 토크는
        # stiffness * (drive_target - actual)이다. 따라서 실제 관절이 원래
        # 목표에 있을 때 gravity 토크가 생기도록 tau_g / stiffness를 더한다.
        raw_bias = gravity / self.stiffness
        position_bias = np.clip(raw_bias, -self.max_bias, self.max_bias)
        drive_target = np.clip(
            self.target + position_bias,
            self.lower_limits,
            self.upper_limits,
        ).astype(np.float32)

        self.arm.go_joints(drive_target)

        self.last_gravity_torque = gravity.astype(np.float32, copy=True)
        self.last_position_bias = (drive_target - self.target).astype(
            np.float32, copy=True
        )
        self.last_drive_target = drive_target.copy()
