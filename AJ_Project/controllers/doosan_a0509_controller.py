"""
doosan_a0509_controller.py
두산 A0509 로봇팔 제어 모듈 (재사용 라이브러리).

run.py는 이 클래스를 불러다 쓰기만 하면 되고, 두산 관련 저수준 처리
(에셋 로드 · 스폰 · DOF 위치제어 · ikpy 역기구학)는 전부 여기 안에 숨긴다.

핵심 개념:
  - 제어 단위는 '관절(DOF)'. 관절각을 주면 링크가 따라 움직인다.
  - go_cartesian(xyz): 좌표를 주면 ikpy IK가 6관절 각도로 변환해 손끝이 그 좌표로.
  - go_joints(q6):     6관절 각도를 직접 준다 (IK 없이).

사용 예:
    arm = DoosanA0509Controller(gym, sim, env, asset_root)
    solved = arm.plan_path([[0.35,0,0.55], [0.30,0.25,0.45], ...])  # 미리 IK
    ...
    arm.go_joints(solved[i])          # 루프에서 목표 관절각 설정
    print(arm.current_tcp())          # 현재 손끝 좌표
의존: ikpy (pip install ikpy==3.3.4)
"""
import os
import numpy as np
from isaacgym import gymapi
from ikpy.chain import Chain


class DoosanA0509Controller:
    def __init__(self, gym, sim, env, asset_root,
                 urdf="urdf/doosan_a0509/a0509.urdf",
                 ik_urdf=None, fix_base=True,
                 spawn_transform=None, actor_name="doosan_a0509",
                 stiffness=600.0, damping=50.0,
                 collision_group=0, collision_filter=1):
        """
        gym/sim/env  : Isaac Gym 핸들 (run.py에서 만들어 넘겨줌)
        asset_root   : 에셋 최상위 경로 (ISAAC_ASSETS)
        urdf         : 액터로 스폰할 URDF (예: 선반+로봇 통합 에셋)
        ik_urdf      : IK 계산용 URDF (None이면 urdf 사용). 통합 에셋을 쓸 때는
                       순수 로봇팔만 든 원본 a0509.urdf를 지정해 IK 체인을 만든다.
        fix_base     : 베이스를 월드에 고정할지 (True=고정, False=바닥에 얹힘)
        spawn_transform : 배치 위치/회전 (기본: 바닥 원점)
        stiffness/damping : DOF 위치제어 PD 게인 (P/D)
        """
        self.gym = gym
        self.sim = sim
        self.env = env
        self.urdf_path = os.path.join(asset_root, ik_urdf or urdf)  # IK 체인용 경로

        # --- (a) 설계도(asset) 준비: URDF → asset ---
        opts = gymapi.AssetOptions()
        opts.fix_base_link = fix_base  # True=월드 고정 / False=바닥에 얹힘
        opts.armature = 0.01           # 관절 관성 보정(수치 안정성)
        self.asset = gym.load_asset(sim, asset_root, urdf, opts)
        self.num_dofs = gym.get_asset_dof_count(self.asset)   # A0509 = 6

        # --- (b) 방(env)에 객체(actor) 배치 ---
        if spawn_transform is None:
            spawn_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        self.actor = gym.create_actor(env, self.asset, spawn_transform,
                                      actor_name, collision_group, collision_filter)

        # --- (c) 관절(DOF) 위치제어 + PD 게인 설정 ---
        props = gym.get_actor_dof_properties(env, self.actor)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)   # 목표각만 주면 PD가 토크 계산
        props["stiffness"].fill(stiffness)             # P게인
        props["damping"].fill(damping)                 # D게인
        gym.set_actor_dof_properties(env, self.actor, props)

        # --- (d) ikpy 체인 준비 (좌표↔관절각 변환용) ---
        #   base_link→joint_1 팔 체인만; 'base' 더미 가지는 제외.
        #   active mask: [Base(fixed)=False, joint_1..joint_6 = True]
        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=["base_link", "joint_1"],
            active_links_mask=[False, True, True, True, True, True, True],
            name="a0509",
        )
        self._last_q = None   # 직전 IK 해 (연속성 seed)

    # ---------------- 역기구학 / 순기구학 ----------------
    def solve_ik(self, target_xyz, seed_6=None):
        """좌표(x,y,z) → 6관절 각도(rad). 반환: (q6, 실제도달좌표, 오차mm)."""
        init = np.zeros(len(self.chain.links))
        if seed_6 is not None:
            init[1:7] = seed_6
        sol = self.chain.inverse_kinematics(target_position=target_xyz,
                                            initial_position=init)
        reached = self.chain.forward_kinematics(sol)[:3, 3]
        err_mm = np.linalg.norm(np.array(target_xyz) - reached) * 1000.0
        return sol[1:7].astype(np.float32), reached, err_mm

    def current_joints(self):
        """현재 6관절 각도(rad)."""
        ds = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_POS)
        return np.array(ds["pos"], dtype=np.float32)

    def current_tcp(self):
        """현재 관절각 → FK → 손끝(link_6) 좌표(x,y,z)."""
        full = np.zeros(len(self.chain.links))
        full[1:7] = self.current_joints()
        return self.chain.forward_kinematics(full)[:3, 3]

    # ---------------- 명령 ----------------
    def go_joints(self, q6):
        """6관절 목표각(rad)을 직접 설정 (IK 없이)."""
        self.gym.set_actor_dof_position_targets(self.env, self.actor,
                                                np.asarray(q6, dtype=np.float32))

    def go_cartesian(self, target_xyz):
        """좌표(x,y,z)를 목표로 설정 (IK로 변환). 직전 해를 seed로 연속성 유지."""
        q6, reached, err = self.solve_ik(target_xyz, seed_6=self._last_q)
        self._last_q = q6
        self.go_joints(q6)
        return reached, err

    def plan_path(self, waypoints, verbose=True):
        """좌표 웨이포인트 리스트를 미리 IK로 풀어 관절각 리스트로 변환.
        직전 해를 seed로 이어 자세 튀는 것을 막는다. 반환: [q6, q6, ...]"""
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
