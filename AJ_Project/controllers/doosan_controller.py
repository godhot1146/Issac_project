"""
doosan_controller.py
두산 A0509 통합 제어 모듈 — JSC / TSC / OSC 를 한 클래스로 제공.

  JSC (관절공간)   : go_joints(q6)          관절각 직접, IK 없음
  TSC (작업공간)   : go_cartesian(xyz)      ikpy IK로 좌표→관절각
  OSC (작업공간·동역학): osc_update(xyz)     자코비안·질량행렬로 토크 계산 + 중력보상

제어 모드가 다르므로(POS ↔ EFFORT) 내부에서 자동 전환한다:
  - JSC/TSC → DOF_MODE_POS (위치제어 PD, stiffness/damping)
  - OSC     → DOF_MODE_EFFORT (토크제어, 내장PD 끔)

torch/gymtorch는 OSC(setup_osc)에서만 지연 import → JSC/TSC만 쓰면 ninja 불필요.
단, OSC는 자코비안·질량행렬이 깔끔하도록 '고정베이스 순수 로봇팔' 액터에서 써야 한다.
"""
import os
import numpy as np
from isaacgym import gymapi
from ikpy.chain import Chain


class DoosanController:
    def __init__(self, gym, sim, env, asset_root,
                 urdf="urdf/doosan_a0509/a0509.urdf", ik_urdf=None,
                 ee_link="link_6", spawn_transform=None, actor_name="a0509",
                 fix_base=True, stiffness=600.0, damping=50.0,
                 kp=150.0, kd=None, gravity_comp=True, g=9.81):
        """
        urdf     : 액터로 스폰할 URDF (통합 에셋 등)
        ik_urdf  : IK 계산용 URDF(None이면 urdf). 통합 에셋일 때 순수 팔 지정.
        fix_base : 베이스 고정 여부 (OSC 쓰려면 반드시 True인 순수 팔 권장)
        stiffness/damping : POS 모드(JSC/TSC) PD 게인
        kp/kd/gravity_comp: OSC 게인 및 중력보상 on/off
        """
        self.gym, self.sim, self.env = gym, sim, env
        self.actor_name = actor_name
        self.stiffness, self.damping = stiffness, damping
        self.kp = kp
        self.kd = kd if kd is not None else 2.0 * np.sqrt(kp)
        self.gravity_comp = gravity_comp
        self.g = g

        # --- 에셋 로드 + 스폰 ---
        opts = gymapi.AssetOptions()
        opts.fix_base_link = fix_base
        opts.armature = 0.01
        self.asset = gym.load_asset(sim, asset_root, urdf, opts)
        self.num_dofs = gym.get_asset_dof_count(self.asset)
        if spawn_transform is None:
            spawn_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        self.actor = gym.create_actor(env, self.asset, spawn_transform, actor_name, 0, 1)

        # --- ikpy 체인 (TSC + FK) ---
        self.chain = Chain.from_urdf_file(
            os.path.join(asset_root, ik_urdf or urdf),
            base_elements=["base_link", "joint_1"],
            active_links_mask=[False, True, True, True, True, True, True],
            name="a0509",
        )
        self._last_q = None

        # --- OSC용 EE 인덱스 ---
        self.ee_index = gym.get_asset_rigid_body_dict(self.asset)[ee_link]

        # 기본은 위치제어(JSC/TSC) 모드
        self._mode = None
        self._set_position_mode()
        self._osc_ready = False

    # ==================== 드라이브 모드 전환 ====================
    def _set_position_mode(self):
        if self._mode == "position":
            return
        p = self.gym.get_actor_dof_properties(self.env, self.actor)
        p["driveMode"].fill(gymapi.DOF_MODE_POS)
        p["stiffness"].fill(self.stiffness)
        p["damping"].fill(self.damping)
        self.gym.set_actor_dof_properties(self.env, self.actor, p)
        self._mode = "position"

    def _set_effort_mode(self):
        if self._mode == "effort":
            return
        p = self.gym.get_actor_dof_properties(self.env, self.actor)
        p["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        p["stiffness"].fill(0.0)
        p["damping"].fill(0.0)
        self.gym.set_actor_dof_properties(self.env, self.actor, p)
        self._mode = "effort"

    # ==================== 기구학 (IK / FK) ====================
    def solve_ik(self, target_xyz, target_R=None, seed_6=None):
        """좌표(+선택적 자세) → 6관절각(rad). 반환: (q6, 도달좌표, 오차mm).
        target_R: 3x3 회전행렬. 주면 손끝 자세까지 구속(orientation_mode='all')."""
        init = np.zeros(len(self.chain.links))
        if seed_6 is not None:
            init[1:7] = seed_6
        if target_R is not None:
            sol = self.chain.inverse_kinematics(
                target_position=target_xyz, target_orientation=np.asarray(target_R),
                orientation_mode="all", initial_position=init)
        else:
            sol = self.chain.inverse_kinematics(target_position=target_xyz, initial_position=init)
        reached = self.chain.forward_kinematics(sol)[:3, 3]
        err_mm = np.linalg.norm(np.array(target_xyz) - reached) * 1000.0
        return sol[1:7].astype(np.float32), reached, err_mm

    def current_joints(self):
        ds = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_POS)
        return np.array(ds["pos"], dtype=np.float32)

    def current_pose(self):
        """현재 손끝 (위치 3, 회전행렬 3x3)."""
        full = np.zeros(len(self.chain.links))
        full[1:7] = self.current_joints()
        T = self.chain.forward_kinematics(full)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def current_tcp(self):
        """현재 관절각 → FK → 손끝 좌표 (모드 무관)."""
        full = np.zeros(len(self.chain.links))
        full[1:7] = self.current_joints()
        return self.chain.forward_kinematics(full)[:3, 3]

    # ==================== JSC (관절공간) ====================
    def go_joints(self, q6):
        """6관절 목표각 직접 설정 (위치제어)."""
        self._set_position_mode()
        self.gym.set_actor_dof_position_targets(self.env, self.actor,
                                                np.asarray(q6, dtype=np.float32))

    # ==================== TSC (작업공간·IK) ====================
    def go_cartesian(self, target_xyz, target_R=None):
        """좌표(+선택적 자세) 목표 → IK → 위치제어. 직전 해를 seed로 연속성.
        target_R을 주면 평행이동 중에도 손끝 자세(그리퍼 절대각)를 유지한다."""
        q6, reached, err = self.solve_ik(target_xyz, target_R=target_R, seed_6=self._last_q)
        self._last_q = q6
        self.go_joints(q6)
        return reached, err

    def plan_path(self, waypoints, verbose=True):
        """좌표 리스트 → 관절각 리스트 사전 IK (seed 이어감)."""
        solved, seed = [], None
        if verbose:
            print("=== A0509 웨이포인트 IK 사전 계산 ===")
        for wp in waypoints:
            q6, reached, err = self.solve_ik(wp, seed_6=seed)
            solved.append(q6)
            seed = q6
            if verbose:
                print(f"  좌표 {wp} → q(deg)={np.round(np.rad2deg(q6),1)}  오차 {err:.2f}mm")
        return solved

    # ==================== OSC (작업공간·동역학) ====================
    def setup_osc(self):
        """prepare_sim() 이후 호출 — 동역학 텐서 확보(지연 import)."""
        import torch                       # isaacgym 이후 import (호출 시점 보장)
        from isaacgym import gymtorch
        self.torch, self.gymtorch = torch, gymtorch
        g, s = self.gym, self.sim
        self._rb  = gymtorch.wrap_tensor(g.acquire_rigid_body_state_tensor(s))
        self._jac = gymtorch.wrap_tensor(g.acquire_jacobian_tensor(s, self.actor_name))
        self._mm  = gymtorch.wrap_tensor(g.acquire_mass_matrix_tensor(s, self.actor_name))
        self._jrow = self.ee_index - 1
        rbp = g.get_actor_rigid_body_properties(self.env, self.actor)
        self._nbody = len(rbp)
        self._masses = torch.tensor([p.mass for p in rbp], dtype=torch.float32)
        self._com = torch.tensor([[p.com.x, p.com.y, p.com.z] for p in rbp], dtype=torch.float32)
        # 전역 rigid-body 텐서에서 이 액터의 바디 시작 인덱스 (다중 액터 씬 대응)
        self._rb_off = g.get_actor_rigid_body_index(self.env, self.actor, 0, gymapi.DOMAIN_SIM)
        self._osc_ready = True

    def _rb_arm(self):
        """전역 rb 텐서에서 이 로봇의 바디들만 슬라이스."""
        return self._rb[self._rb_off:self._rb_off + self._nbody]

    def _quat_rotate(self, q, v):
        t = self.torch
        qv = q[0:3]
        tt = 2.0 * t.linalg.cross(qv, v)
        return v + q[3] * tt + t.linalg.cross(qv, tt)

    def _gravity_torque(self):
        """중력보상 토크 τ_g[j] = Σ_i m_i·g·∂z_com_i/∂q_j."""
        t = self.torch
        tau = t.zeros(self.num_dofs)
        quats = self._rb_arm()[:, 3:7]
        for i in range(1, self._nbody):        # base_link(0) 제외
            m = float(self._masses[i])
            if m <= 0.0:
                continue
            r = self._quat_rotate(quats[i], self._com[i])
            Jl = self._jac[0, i - 1, 0:3, :]
            Jw = self._jac[0, i - 1, 3:6, :]
            Jcom = Jl + t.linalg.cross(Jw, r.unsqueeze(1).expand(3, self.num_dofs), dim=0)
            tau += m * self.g * Jcom[2, :]
        return tau

    def _osc_torque(self, target_xyz):
        t = self.torch
        g, s = self.gym, self.sim
        g.refresh_rigid_body_state_tensor(s)
        g.refresh_jacobian_tensors(s)
        g.refresh_mass_matrix_tensors(s)

        rb = self._rb_arm()
        ee_pos = rb[self.ee_index, 0:3]
        ee_vel = rb[self.ee_index, 7:10]
        j_pos  = self._jac[0, self._jrow, 0:3, :]
        mm     = self._mm[0]

        lam = t.inverse(j_pos @ t.inverse(mm) @ j_pos.t())
        tgt = t.as_tensor(target_xyz, dtype=ee_pos.dtype)
        pos_err = tgt - ee_pos
        F = lam @ (self.kp * pos_err - self.kd * ee_vel)
        tau = j_pos.t() @ F
        if self.gravity_comp:
            tau = tau + self._gravity_torque()
        return tau, ee_pos.clone(), float(t.norm(pos_err))

    def osc_update(self, target_xyz):
        """OSC 1스텝: 토크 계산+인가. 반환: (ee_pos, err_m). 매 프레임 호출."""
        if not self._osc_ready:
            raise RuntimeError("osc_update 전에 setup_osc()를 prepare_sim 이후 호출해야 함")
        self._set_effort_mode()
        tau, ee_pos, err = self._osc_torque(target_xyz)
        self.gym.set_dof_actuation_force_tensor(
            self.sim, self.gymtorch.unwrap_tensor(tau.contiguous()))
        return ee_pos, err
