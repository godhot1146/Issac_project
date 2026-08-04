import torch

from indy7_controller_v1 import (
    IndyArmController,
    ArmMotionStep,      
    ArmStepRunner,       
    JointMotionGenerator,
    LinearMotionGenerator,
)


class IndyArmWithSubArmController(IndyArmController):
    """
    sub_arm_joint / cylinder_joint가 추가된 새 URDF용 컨트롤러.
    나머지(IK 좌표 조작, 키보드 수동 조작, attach/detach 등)는 전부 부모 그대로 사용한다.
    """

    def __init__(self, gym, sim, env, viewer, asset,
                 spawn_transform, actor_name="indy7_asset",
                 stiffness=50000.0, damping_gain=1000.0,
                 ik_damping=0.1,
                 move_speed=0.01, rot_speed=0.05,
                 linear_speed=5.0, angular_speed=1.0, max_joint_vel=None,
                 lock_weight=3000.0, free_weight=1.0,
                 dt=1.0 / 60.0,
                 sub_arm_max_vel=1.0, cylinder_max_vel=0.2,
                 sub_arm_effort=200.0, cylinder_effort=200.0):

        # DOF 순서상 기대되는 인덱스(6, 7). 실제 인덱스는 setup_tensors()에서
        # dof_names로 재확인해서 덮어쓴다 (에셋 파싱 순서가 다를 가능성 대비).
        self.SUB_ARM_DOF_IDX = 6
        self.CYLINDER_DOF_IDX = 7
        self._sub_arm_max_vel = sub_arm_max_vel
        self._cylinder_max_vel = cylinder_max_vel

        # 부모가 max_joint_vel=None일 때 6-DOF 기본값을 만들어버리므로,
        # 여기서 sub_arm_joint/cylinder_joint 몫까지 포함한 8-DOF 기본값을 미리 구성해서 넘긴다.
        if max_joint_vel is None:
            max_joint_vel = torch.tensor([
                2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159,  # joint0~5 (기존과 동일)
                sub_arm_max_vel,                                        # sub_arm_joint
                cylinder_max_vel,                                       # cylinder_joint
            ])

        super().__init__(
            gym=gym, sim=sim, env=env, viewer=viewer, asset=asset,
            spawn_transform=spawn_transform, actor_name=actor_name,
            stiffness=stiffness, damping_gain=damping_gain, ik_damping=ik_damping,
            move_speed=move_speed, rot_speed=rot_speed,
            linear_speed=linear_speed, angular_speed=angular_speed,
            max_joint_vel=max_joint_vel,
            lock_weight=lock_weight, free_weight=free_weight, dt=dt,
        )

        # --- ee를 body_names[-1](=sub_arm_part)이 아니라 vacuum으로 고정 ---
        # 부모 __init__이 self.ee_name = body_names[-1] 로 이미 설정해둔 것을 덮어쓴다.
        # setup_tensors()가 실행될 때(외부에서 별도 호출) 이 이름 기준으로
        # ee_index_actor / ee_index_global이 계산되므로, 그 전에만 바꿔주면 된다.
        self.ee_name = "vacuum"
        print(f"[IndyArmWithSubArm] end-effector를 '{self.ee_name}'(으)로 고정")

        # --- 새로 추가된 DOF(6: sub_arm_joint, 7: cylinder_joint)의 effort 별도 지정 ---
        # 부모는 인덱스 0~5까지만 effort를 세팅하므로, 여기서 나머지를 채운다.
        dof_props = gym.get_actor_dof_properties(env, self.handle)
        if self.dof_count > self.SUB_ARM_DOF_IDX:
            dof_props["effort"][self.SUB_ARM_DOF_IDX] = sub_arm_effort
        if self.dof_count > self.CYLINDER_DOF_IDX:
            dof_props["effort"][self.CYLINDER_DOF_IDX] = cylinder_effort
        gym.set_actor_dof_properties(env, self.handle, dof_props)

    # ==========================================================================
    # setup_tensors: 부모 로직 실행 후 sub_arm/cylinder DOF 인덱스를 이름으로 재확인
    # ==========================================================================
    def setup_tensors(self):
        super().setup_tensors()

        try:
            self.SUB_ARM_DOF_IDX = self.dof_names.index("sub_arm_joint")
        except ValueError:
            self.SUB_ARM_DOF_IDX = None
            print("[IndyArmWithSubArm] 경고: DOF 목록에서 'sub_arm_joint'를 찾지 못했습니다.")

        try:
            self.CYLINDER_DOF_IDX = self.dof_names.index("cylinder_joint")
        except ValueError:
            self.CYLINDER_DOF_IDX = None
            print("[IndyArmWithSubArm] 경고: DOF 목록에서 'cylinder_joint'를 찾지 못했습니다.")

        print(f"[IndyArmWithSubArm] sub_arm_joint DOF idx = {self.SUB_ARM_DOF_IDX}, "
              f"cylinder_joint DOF idx = {self.CYLINDER_DOF_IDX}")

    # ==========================================================================
    # move_to_joint_pose 오버라이드: callable(self) -> tensor 도 허용
    #   -> ArmStepRunner가 스텝을 '실행하는 시점'의 현재 관절각을 기준으로
    #      목표를 동적으로 계산할 수 있음 (sub_arm_delta_target 등에 사용).
    #   ArmStepRunner._start()의 `self.arm.move_to_joint_pose(step.target)` 호출은
    #   그대로 두고, 여기서만 target이 함수면 실행해서 tensor로 바꿔준다.
    # ==========================================================================
    def move_to_joint_pose(self, name_or_tensor, max_joint_vel=None, synchronized=True):
        if callable(name_or_tensor) and not isinstance(name_or_tensor, str):
            name_or_tensor = name_or_tensor(self)
        super().move_to_joint_pose(name_or_tensor, max_joint_vel=max_joint_vel,
                                    synchronized=synchronized)

    # ==========================================================================
    # 6-DOF(팔) 각도만으로 8-DOF 전체 목표 벡터 구성
    #   -> 기존에 등록해둔 6-DOF 프리셋을 그대로 재사용하고 싶을 때 사용
    # ==========================================================================
    def build_full_dof_target(self, arm_q, sub_arm_angle=None, cylinder_pos=None):
        """
        arm_q: 길이 6 (joint0~joint5) 텐서/리스트.
        sub_arm_angle, cylinder_pos: None이면 '현재 목표값(직전 관절 목표)'을 그대로 유지.
        반환: 길이 dof_count(=8) torch.Tensor. move_to_joint_pose()에 바로 넘길 수 있다.
        """
        device = self.dof_states_full.device
        if not torch.is_tensor(arm_q):
            arm_q = torch.tensor(arm_q, dtype=torch.float32, device=device)
        arm_q = arm_q.to(device)

        self.gym.refresh_dof_state_tensor(self.sim)
        full = self.dof_states_full[self.dof_indices, 0].clone()
        full[:6] = arm_q
        if sub_arm_angle is not None and self.SUB_ARM_DOF_IDX is not None:
            full[self.SUB_ARM_DOF_IDX] = float(sub_arm_angle)
        if cylinder_pos is not None and self.CYLINDER_DOF_IDX is not None:
            full[self.CYLINDER_DOF_IDX] = float(cylinder_pos)
        return full

    def register_joint_pose(self, name, angles_rad):
        """
        부모와 달리 길이 6(팔 관절만) 또는 길이 dof_count(=8, 팔+sub_arm+cylinder)
        둘 다 등록 가능하도록 확장. 길이 6이면 등록 시점의 sub_arm/cylinder 목표값을
        그대로 붙여서 8-DOF로 저장한다.
        """
        device = self.dof_states_full.device
        t = torch.tensor(angles_rad, dtype=torch.float32, device=device)
        if t.numel() == self.dof_count:
            self.joint_poses[name] = t
        elif t.numel() == 6:
            self.joint_poses[name] = self.build_full_dof_target(t)
        else:
            raise ValueError(
                f"register_joint_pose('{name}'): 길이 {t.numel()}는 지원하지 않습니다 "
                f"(6 또는 {self.dof_count}만 허용)"
            )

    # ==========================================================================
    # sub_arm_joint 전용 헬퍼
    # ==========================================================================
    def sub_arm_abs_target(self, angle_rad):
        """
        ArmMotionStep(kind='joint', target=...)에 그대로 꽂을 수 있는 콜러블 반환.
        스텝이 '실행되는 시점'의 팔/실린더 목표는 그대로 두고, sub_arm_joint만
        절대각(angle_rad)으로 이동시킨다.
        """
        def _target(arm):
            if arm.SUB_ARM_DOF_IDX is None:
                raise RuntimeError("sub_arm_joint DOF를 찾을 수 없습니다.")
            arm.gym.refresh_dof_state_tensor(arm.sim)
            cur = arm.dof_states_full[arm.dof_indices, 0].clone()
            cur[arm.SUB_ARM_DOF_IDX] = angle_rad
            return cur
        return _target

    def sub_arm_delta_target(self, delta_rad):
        """현재 sub_arm_joint 각도 기준 상대 회전(delta_rad)으로 이동시키는 콜러블 반환."""
        def _target(arm):
            if arm.SUB_ARM_DOF_IDX is None:
                raise RuntimeError("sub_arm_joint DOF를 찾을 수 없습니다.")
            arm.gym.refresh_dof_state_tensor(arm.sim)
            cur = arm.dof_states_full[arm.dof_indices, 0].clone()
            cur[arm.SUB_ARM_DOF_IDX] += delta_rad
            return cur
        return _target

    def move_sub_arm_to(self, angle_rad, synchronized=True):
        """시퀀스 밖에서 즉시 sub_arm_joint를 절대각으로 이동시키고 싶을 때."""
        self.move_to_joint_pose(self.sub_arm_abs_target(angle_rad), synchronized=synchronized)

    def move_sub_arm_by(self, delta_rad, synchronized=True):
        """시퀀스 밖에서 즉시 sub_arm_joint를 상대각만큼 이동시키고 싶을 때."""
        self.move_to_joint_pose(self.sub_arm_delta_target(delta_rad), synchronized=synchronized)

    # ==========================================================================
    # cylinder_joint(prismatic) 전용 헬퍼 (필요 시 sub_arm과 동일한 패턴으로 사용)
    # ==========================================================================
    def cylinder_abs_target(self, pos):
        def _target(arm):
            if arm.CYLINDER_DOF_IDX is None:
                raise RuntimeError("cylinder_joint DOF를 찾을 수 없습니다.")
            arm.gym.refresh_dof_state_tensor(arm.sim)
            cur = arm.dof_states_full[arm.dof_indices, 0].clone()
            cur[arm.CYLINDER_DOF_IDX] = pos
            return cur
        return _target

    def move_cylinder_to(self, pos, synchronized=True):
        self.move_to_joint_pose(self.cylinder_abs_target(pos), synchronized=synchronized)
