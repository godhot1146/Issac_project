"""볶음 자동 시퀀스 전용 중력보상 관절 제어.

공용 ``DoosanController``는 수정하지 않는다. 대신 이미 ``setup_osc()``가
확보한 동역학 텐서와 중력 토크 계산을 재사용해, 자동 운전 중에만 관절
effort 제어를 적용한다.
"""

import numpy as np
from isaacgym import gymapi


class StirfryAutoJointControl:
    """중력보상 계산토크로 6축 목표각을 매 물리 프레임 추종한다."""

    def __init__(self, arm, kp=150.0, kd=None):
        if not arm._osc_ready:
            raise RuntimeError(
                "StirfryAutoJointControl 생성 전에 arm.setup_osc()가 필요합니다."
            )

        self.arm = arm
        self.gym = arm.gym
        self.sim = arm.sim
        self.env = arm.env
        self.kp = self._gain_array(kp, "kp")
        if kd is None:
            kd = 2.0 * np.sqrt(self.kp)
        self.kd = self._gain_array(kd, "kd")

        dof_properties = self.gym.get_actor_dof_properties(
            self.env, self.arm.actor
        )
        self.effort_limits = np.asarray(
            dof_properties["effort"], dtype=np.float32
        ).copy()
        if (
            self.effort_limits.shape != (self.arm.num_dofs,)
            or not np.all(np.isfinite(self.effort_limits))
            or np.any(self.effort_limits <= 0.0)
        ):
            raise RuntimeError(
                "자동 중력보상에 필요한 관절 effort limit을 URDF에서 읽지 못했습니다."
            )

        self.target = None
        self.last_gravity_torque = np.zeros(self.arm.num_dofs, dtype=np.float32)
        self.last_command_torque = np.zeros(self.arm.num_dofs, dtype=np.float32)

        print(
            "[자동 제어] 중력보상 계산토크 활성화 "
            f"(Kp={self.kp[0]:.0f}, Kd={self.kd[0]:.1f}, "
            f"effort limit={np.round(self.effort_limits, 1)} Nm)"
        )

    def _gain_array(self, value, name):
        gain = np.asarray(value, dtype=np.float32)
        if gain.ndim == 0:
            gain = np.full(self.arm.num_dofs, float(gain), dtype=np.float32)
        if (
            gain.shape != (self.arm.num_dofs,)
            or not np.all(np.isfinite(gain))
            or np.any(gain < 0.0)
        ):
            raise ValueError(f"{name}는 음수가 아닌 {self.arm.num_dofs}축 값이어야 합니다.")
        return gain

    def command(self, target_joints):
        """새 목표각을 저장하고 현재 물리 프레임의 토크를 인가한다."""
        target = np.asarray(target_joints, dtype=np.float32)
        if target.shape != (self.arm.num_dofs,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"target_joints는 유효한 {self.arm.num_dofs}축 값이어야 합니다."
            )
        self.target = target.copy()
        self.update()

    def capture_current_target(self):
        """실패 지점에서 더 밀지 않도록 현재 관절각을 새 유지 목표로 삼는다."""
        self.target = self.arm.current_joints().astype(np.float32)

    def update(self):
        """저장한 목표각을 중력보상 effort로 한 프레임 유지한다."""
        if self.target is None:
            raise RuntimeError("자동 관절 목표가 설정되지 않았습니다.")

        self.arm._set_effort_mode()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        states = self.gym.get_actor_dof_states(
            self.env, self.arm.actor, gymapi.STATE_ALL
        )
        q = np.asarray(states["pos"], dtype=np.float32)
        qd = np.asarray(states["vel"], dtype=np.float32)

        gravity = self.arm._gravity_torque()
        torch = self.arm.torch
        device = gravity.device
        dtype = gravity.dtype
        q_tensor = torch.as_tensor(q, dtype=dtype, device=device)
        qd_tensor = torch.as_tensor(qd, dtype=dtype, device=device)
        target_tensor = torch.as_tensor(self.target, dtype=dtype, device=device)
        kp_tensor = torch.as_tensor(self.kp, dtype=dtype, device=device)
        kd_tensor = torch.as_tensor(self.kd, dtype=dtype, device=device)
        limits = torch.as_tensor(
            self.effort_limits, dtype=dtype, device=device
        )

        desired_acceleration = (
            kp_tensor * (target_tensor - q_tensor) - kd_tensor * qd_tensor
        )
        # 질량행렬을 곱한 계산토크는 같은 게인을 raw PD 토크로 쓰는 것보다
        # 관절별 관성 차이에 덜 민감하고, 느린 접촉 경로에서 안정적이다.
        torque = self.arm._mm[0] @ desired_acceleration + gravity
        torque = torch.maximum(torch.minimum(torque, limits), -limits)
        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            self.arm.gymtorch.unwrap_tensor(torque.contiguous()),
        )

        self.last_gravity_torque = gravity.detach().cpu().numpy().copy()
        self.last_command_torque = torque.detach().cpu().numpy().copy()
